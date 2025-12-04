const std = @import("std");

const CONDA_ENV = "DeepReadMapper";
const ZIG_OUT = "zig-out";
const BIN_OUT = ZIG_OUT ++ "/bin";
const OBJ_OUT = ZIG_OUT ++ "/obj";

/// Convert source path to guaranteed-unique object file name using hash
fn sourceToObjectName(builder: *std.Build, source_path: []const u8) []const u8 {
    // Create a hash of the full source path
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source_path, &digest, .{});

    // Convert to hex string
    const short_slice = digest[0..16]; // Use first 16 bytes -> 32 hex chars
    const hex_hash = std.fmt.bytesToHex(short_slice, .lower);
    // Get the base filename (without extension) for readability
    const basename = std.fs.path.stem(std.fs.path.basename(source_path));

    // Combine: {basename}_{hash}
    return builder.fmt("{s}_{s}", .{ basename, hex_hash });
}

/// Function to compile source to object file with dependency checking
fn createObjectCmd(builder: *std.Build, compiler_path: []const u8, source: []const u8, conda_prefix: []const u8, common_flags: []const []const u8, common_includes: []const []const u8) *std.Build.Step.Run {
    const cmd = builder.addSystemCommand(&[_][]const u8{compiler_path});

    for (common_flags) |flag| {
        cmd.addArg(flag);
    }

    // Compile only
    cmd.addArg("-c");

    for (common_includes) |inc| {
        cmd.addArg(inc);
    }

    cmd.addArg(builder.fmt("-isystem{s}/include", .{conda_prefix}));
    cmd.addArg(source);

    // Get object file name based on source path
    const obj_name = sourceToObjectName(builder, source);
    cmd.addArgs(&[_][]const u8{ "-o", builder.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name }) });

    // Add dependency file generation for incremental builds
    cmd.addArg("-MMD");
    cmd.addArg("-MP");
    cmd.addArg(builder.fmt("-MF{s}/{s}.d", .{ OBJ_OUT, obj_name }));

    return cmd;
}

/// Check if object file needs rebuilding based on timestamps
/// Returns true if:
/// 1. Source file does not exist
/// 2. Object file or dependency file does not exist
/// 3. Source file is newer than object file
/// 4. Any header dependency is newer than object file
fn needsRebuild(builder: *std.Build, source_path: []const u8) bool {
    // Get object file name based on source path
    const obj_name = sourceToObjectName(builder, source_path);

    // Check if object file exists
    const obj_path = builder.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name });
    const obj_stat = std.fs.cwd().statFile(obj_path) catch return true;

    // Check source file timestamp against object file
    const src_stat = std.fs.cwd().statFile(source_path) catch std.debug.panic("Source file not found: {s}", .{source_path});

    // If source is newer, need to rebuild
    if (src_stat.mtime > obj_stat.mtime) {
        return true;
    }

    // Check dependency file for included headers
    const dep_path = builder.fmt("{s}/{s}.d", .{ OBJ_OUT, obj_name });
    _ = std.fs.cwd().statFile(dep_path) catch return true;

    // Parse dependency file if it exists
    // std.debug.print("Checking dependencies for {s}\n", .{source_path});
    return checkDependencyTimestamps(builder.allocator, dep_path, obj_stat.mtime);
}

/// Parse .d file and check if any dependency is newer than object file
fn checkDependencyTimestamps(allocator: std.mem.Allocator, dep_path: []const u8, obj_mtime: i128) bool {
    const file = std.fs.cwd().openFile(dep_path, .{}) catch return true; // Force rebuild on open failure
    defer file.close();

    // Take a reader
    var reader_buf: [1024]u8 = undefined;
    var reader = file.reader(&reader_buf);

    // Buffer to hold logical line
    var logical_line_buf = std.array_list.Managed(u8).init(allocator);
    defer logical_line_buf.deinit();

    // Flag to check if last char was backslash
    var last_was_backslash = false;

    while (true) {
        const line_buf = reader.interface.take(1024) catch |err| switch (err) {
            error.EndOfStream => break,
            error.ReadFailed => return true, // Force rebuild on read failure
        };
        std.debug.print("Read line: {s}\n", .{line_buf});
        for (line_buf) |b| {
            if (last_was_backslash) {
                // Expect newline
                if (b == '\n') {
                    // Skip both: do NOT append either char
                    last_was_backslash = false;
                    continue;
                } else {
                    std.debug.panic("Invalid .d file: '\\' not followed by newline", .{});
                }
            }

            if (b == '\\') {
                last_was_backslash = true;
                continue;
            }

            if (b == '\n') {
                // normal newline terminates the logical line
                break;
            }
            logical_line_buf.append(b) catch @panic("OOM");
        }
    }

    // Get the logical line
    const line = logical_line_buf.items;

    // Split by whitespace
    var it = std.mem.tokenizeAny(u8, line, " \t\r\n");

    while (it.next()) |tok| {
        // Check if token is at the end of the line
        if (std.mem.endsWith(u8, tok, ":")) {
            continue; // Skip target
        }

        // This is a dependency file - check its timestamp
        const dep_stat = std.fs.cwd().statFile(tok) catch {
            // If dependency file doesn't exist, force rebuild
            return true;
        };

        if (dep_stat.mtime > obj_mtime) {
            // Dependency is newer than object file, need to rebuild
            return true;
        }
    }

    return false; // No dependencies are newer
}

/// Create a conditional object compilation step
fn createConditionalObjectCmd(builder: *std.Build, compiler_path: []const u8, source: []const u8, conda_prefix: []const u8, common_flags: []const []const u8, common_includes: []const []const u8) ?*std.Build.Step.Run {
    // std.debug.print("Checking if {s} needs rebuild...\n", .{source});
    // Only create compilation step if rebuild is needed
    if (!needsRebuild(builder, source)) {
        std.debug.print("Skipping {s} (up to date)\n", .{source});
        return null;
    }

    std.debug.print("Compiling {s}\n", .{source});
    return createObjectCmd(builder, compiler_path, source, conda_prefix, common_flags, common_includes);
}

fn createLinkCmd(builder: *std.Build, compiler_path: []const u8, all_object_files: []const []const u8, output_name: []const u8, extra_libs: []const []const u8, conda_prefix: []const u8) *std.Build.Step.Run {
    const cmd = builder.addSystemCommand(&[_][]const u8{compiler_path});

    cmd.addArgs(&[_][]const u8{
        "-fopenmp",
    });

    for (all_object_files) |obj| {
        cmd.addArg(obj);
    }

    cmd.addArg(builder.fmt("-L{s}/lib", .{conda_prefix}));
    cmd.addArgs(&[_][]const u8{
        "-lstdc++",
        "-lz",
    });

    for (extra_libs) |lib| {
        cmd.addArg(lib);
    }

    cmd.addArgs(&[_][]const u8{ "-o", builder.fmt("{s}/{s}", .{ BIN_OUT, output_name }) });

    return cmd;
}

pub fn build(b: *std.Build) void {
    const conda_prefix = std.posix.getenv("CONDA_PREFIX") orelse {
        std.debug.print("Please activate conda environment: {s}\n", .{CONDA_ENV});
        std.process.exit(1);
    };

    if (std.mem.indexOf(u8, conda_prefix, CONDA_ENV) == null) {
        std.debug.print("Incorrect conda environment: {s}\n", .{conda_prefix});
        std.debug.print("Please activate conda environment: {s}\n", .{CONDA_ENV});
        std.process.exit(1);
    }

    std.debug.print("Using conda prefix: {s}\n", .{conda_prefix});

    const gcc_path = b.findProgram(&[_][]const u8{"g++"}, &[_][]const u8{
        b.fmt("{s}/bin", .{conda_prefix}),
        "/usr/bin",
    }) catch "g++";

    std.debug.print("Using g++ path: {s}\n", .{gcc_path});

    const common_sources = [_][]const u8{
        "external/cnpy/cnpy.cpp",
        "src/utils/parse_inputs.cpp",
        "src/utils/utils.cpp",
        "src/utils/tok2index.cpp",
        "src/utils/post_processor.cpp",
        "src/utils/metrics.cpp",
        "src/utils/reranker.cpp",
        "src/inference/vectorize.cpp",
        "src/inference/fast_model.cpp",
        "src/inference/preprocess.cpp",
    };

    const common_flags = [_][]const u8{
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-march=native",
        "-Wall",
    };

    const common_includes = [_][]const u8{
        "-Iincludes",
        "-Iincludes/inference",
        "-Iincludes/utils",
        "-Iincludes/hnswlib_dir",
        "-Iincludes/gann_hnsw",
        "-Iincludes/hnswpq",
        "-Iexternal/cereal/include",
        "-Iexternal/cnpy",
    };

    const mkdir_bin_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", BIN_OUT });
    const mkdir_obj_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", OBJ_OUT });

    var all_object_files = std.array_list.Managed([]const u8).init(b.allocator);
    var all_object_steps = std.array_list.Managed(*std.Build.Step.Run).init(b.allocator);

    // Compile common sources with incremental checking
    for (common_sources) |source| {
        const obj_path = b.fmt("{s}/{s}.o", .{ OBJ_OUT, sourceToObjectName(b, source) });
        all_object_files.append(obj_path) catch @panic("OOM");

        if (createConditionalObjectCmd(b, gcc_path, source, conda_prefix, &common_flags, &common_includes)) |obj_cmd| {
            obj_cmd.step.dependOn(&mkdir_obj_cmd.step);
            all_object_steps.append(obj_cmd) catch @panic("OOM");
        }
    }

    // Helper function to build executable with incremental compilation
    const BuildExecutable = struct {
        fn build(
            builder: *std.Build,
            gcc: []const u8,
            main_sources: []const []const u8,
            common_objs: []const []const u8,
            common_obj_steps: []const *std.Build.Step.Run,
            output: []const u8,
            libs: []const []const u8,
            prefix: []const u8,
            flags: []const []const u8,
            includes: []const []const u8,
            mkdir_bin: *std.Build.Step.Run,
            mkdir_obj: *std.Build.Step.Run,
        ) *std.Build.Step.Run {
            var exe_objects = std.array_list.Managed([]const u8).init(builder.allocator);
            var exe_obj_steps = std.array_list.Managed(*std.Build.Step.Run).init(builder.allocator);

            // Add common objects
            for (common_objs) |obj| {
                exe_objects.append(obj) catch @panic("OOM");
            }

            // Compile main sources with incremental checking
            for (main_sources) |source| {
                const obj_path = builder.fmt("{s}/{s}.o", .{ OBJ_OUT, sourceToObjectName(builder, source) });
                exe_objects.append(obj_path) catch @panic("OOM");

                if (createConditionalObjectCmd(builder, gcc, source, prefix, flags, includes)) |obj_cmd| {
                    obj_cmd.step.dependOn(&mkdir_obj.step);
                    exe_obj_steps.append(obj_cmd) catch @panic("OOM");
                }
            }

            const link_cmd = createLinkCmd(builder, gcc, exe_objects.items, output, libs, prefix);
            link_cmd.step.dependOn(&mkdir_bin.step);

            // Depend on object compilations
            for (common_obj_steps) |obj_step| {
                link_cmd.step.dependOn(&obj_step.step);
            }
            for (exe_obj_steps.items) |obj_step| {
                link_cmd.step.dependOn(&obj_step.step);
            }

            return link_cmd;
        }
    };

    // Build executables
    const pipeline_cmd = BuildExecutable.build(b, gcc_path, &[_][]const u8{ "src/main.cpp", "src/hnswpq/search.cpp" }, all_object_files.items, all_object_steps.items, "pipeline", &[_][]const u8{ "-lopenvino", "-lomp", "-lfaiss" }, conda_prefix, &common_flags, &common_includes, mkdir_bin_cmd, mkdir_obj_cmd);

    const inference_cmd = BuildExecutable.build(b, gcc_path, &[_][]const u8{"src/inference/test_inference.cpp"}, all_object_files.items, all_object_steps.items, "inference", &[_][]const u8{"-lopenvino"}, conda_prefix, &common_flags, &common_includes, mkdir_bin_cmd, mkdir_obj_cmd);

    const hnswpq_cmd = BuildExecutable.build(b, gcc_path, &[_][]const u8{"src/hnswpq/index.cpp"}, all_object_files.items, all_object_steps.items, "hnswpq_index", &[_][]const u8{ "-lopenvino", "-lomp", "-lfaiss" }, conda_prefix, &common_flags, &common_includes, mkdir_bin_cmd, mkdir_obj_cmd);

    // Build steps
    const install_step = b.getInstallStep();
    install_step.dependOn(&pipeline_cmd.step);
    install_step.dependOn(&inference_cmd.step);
    install_step.dependOn(&hnswpq_cmd.step);

    const pipeline_step = b.step("pipeline", "Build only the pipeline executable");
    pipeline_step.dependOn(&pipeline_cmd.step);

    const inference_step = b.step("inference", "Build only the inference executable");
    inference_step.dependOn(&inference_cmd.step);

    const index_step = b.step("index", "Build only the index executable");
    index_step.dependOn(&hnswpq_cmd.step);

    const objects_step = b.step("objects", "Build common object files");
    for (all_object_steps.items) |obj_step| {
        objects_step.dependOn(&obj_step.step);
    }

    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", BIN_OUT, OBJ_OUT });
    clean_step.dependOn(&clean_cmd.step);
}
