const std = @import("std");

/// Expected conda environment name that should be activated
const CONDA_ENV = "DeepReadMapper";
/// Highest-level build output directory
const ZIG_OUT = "zig-out";
/// Output directory for binaries
const BIN_OUT = ZIG_OUT ++ "/bin";
/// Output directory for object files and dependency files
const OBJ_OUT = ZIG_OUT ++ "/obj";

/// Builder struct that encapsulates all build functionality
const Builder = struct {
    /// Zig build instance
    builder: *std.Build,
    /// Path to g++ compiler
    gxx_path: []const u8,
    /// Conda environment prefix path
    conda_prefix: []const u8,
    /// Command step to create binary output directory
    mkdir_bin_cmd: *std.Build.Step.Run,
    /// Command step to create object output directory
    mkdir_obj_cmd: *std.Build.Step.Run,

    const Self = @This();

    /// Initialize a new Builder instance
    pub fn init(builder: *std.Build, gxx_path: []const u8, conda_prefix: []const u8) Self {
        const mkdir_bin_cmd = builder.addSystemCommand(&[_][]const u8{ "mkdir", "-p", BIN_OUT });
        const mkdir_obj_cmd = builder.addSystemCommand(&[_][]const u8{ "mkdir", "-p", OBJ_OUT });

        return Self{
            .builder = builder,
            .gxx_path = gxx_path,
            .conda_prefix = conda_prefix,
            .mkdir_bin_cmd = mkdir_bin_cmd,
            .mkdir_obj_cmd = mkdir_obj_cmd,
        };
    }

    /// Convert source path to guaranteed-unique object file name using hash (without extension)
    ///
    /// Format: {basename}_{hash}
    fn sourceToObjectName(self: Self, source_path: []const u8) []const u8 {
        // Create a hash of the full source path
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(source_path, &digest, .{});

        // Convert to hex string
        const short_slice = digest[0..16]; // Use first 16 bytes -> 32 hex chars
        const hex_hash = std.fmt.bytesToHex(short_slice, .lower);
        // Get the base filename (without extension) for readability
        const basename = std.fs.path.stem(std.fs.path.basename(source_path));

        // Combine: {basename}_{hash}
        return self.builder.fmt("{s}_{s}", .{ basename, hex_hash });
    }

    /// Function to compile source to object file with dependency checking
    fn createObjectCmd(self: Self, source: []const u8, flags: []const []const u8, includes: []const []const u8) *std.Build.Step.Run {
        const cmd = self.builder.addSystemCommand(&[_][]const u8{self.gxx_path});

        for (flags) |flag| {
            cmd.addArg(flag);
        }

        // Compile only
        cmd.addArg("-c");

        for (includes) |inc| {
            cmd.addArg(inc);
        }

        cmd.addArg(self.builder.fmt("-isystem{s}/include", .{self.conda_prefix}));
        cmd.addArg(source);

        // Get object file name based on source path
        const obj_name = self.sourceToObjectName(source);
        cmd.addArgs(&[_][]const u8{ "-o", self.builder.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name }) });

        // Add dependency file generation for incremental builds
        cmd.addArg("-MMD");
        cmd.addArg("-MP");
        cmd.addArg(self.builder.fmt("-MF{s}/{s}.d", .{ OBJ_OUT, obj_name }));

        return cmd;
    }

    /// Check if object file needs rebuilding based on timestamps
    ///
    /// Returns true if:
    /// 1. Source file does not exist
    /// 2. Object file or dependency file does not exist
    /// 3. Source file is newer than object file
    /// 4. Any header dependency is newer than object file
    fn needsRebuild(self: Self, source_path: []const u8) bool {
        // Get object file name based on source path
        const obj_name = self.sourceToObjectName(source_path);

        // Check if object file exists
        const obj_path = self.builder.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name });
        const obj_stat = std.fs.cwd().statFile(obj_path) catch return true;

        // Check source file timestamp against object file
        const src_stat = std.fs.cwd().statFile(source_path) catch std.debug.panic("Source file not found: {s}", .{source_path});

        // If source is newer, need to rebuild
        if (src_stat.mtime > obj_stat.mtime) {
            return true;
        }

        // Check dependency file for included headers
        const dep_path = self.builder.fmt("{s}/{s}.d", .{ OBJ_OUT, obj_name });
        _ = std.fs.cwd().statFile(dep_path) catch return true;

        // Parse dependency file if it exists
        return self.checkDependencyTimestamps(dep_path, obj_stat.mtime);
    }

    /// Parse .d file and check if any dependency is newer than object file
    fn checkDependencyTimestamps(self: Self, dep_path: []const u8, obj_mtime: i128) bool {
        const file = std.fs.cwd().openFile(dep_path, .{}) catch return true; // Force rebuild on open failure
        defer file.close();

        // Take a reader
        var reader_buf: [1024]u8 = undefined;
        var reader = file.reader(&reader_buf);

        // Buffer to hold logical line
        var logical_line_buf = std.array_list.Managed(u8).init(self.builder.allocator);
        defer logical_line_buf.deinit();

        // Flag to check if last char was backslash
        var last_was_backslash = false;

        while (true) {
            const line_buf = reader.interface.take(1024) catch |err| switch (err) {
                error.EndOfStream => break,
                error.ReadFailed => return true, // Force rebuild on read failure
            };

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
    fn createConditionalObjectCmd(self: Self, source: []const u8, flags: []const []const u8, includes: []const []const u8) ?*std.Build.Step.Run {
        // Only create compilation step if rebuild is needed
        if (!self.needsRebuild(source)) {
            std.debug.print("Skipping {s} (up to date)\n", .{source});
            return null;
        }

        std.debug.print("Compiling {s}\n", .{source});
        return self.createObjectCmd(source, flags, includes);
    }

    /// Create linking command
    fn createLinkCmd(self: Self, all_object_files: []const []const u8, output_name: []const u8, extra_libs: []const []const u8) *std.Build.Step.Run {
        const cmd = self.builder.addSystemCommand(&[_][]const u8{self.gxx_path});

        cmd.addArgs(&[_][]const u8{
            "-fopenmp",
        });

        for (all_object_files) |obj| {
            cmd.addArg(obj);
        }

        cmd.addArg(self.builder.fmt("-L{s}/lib", .{self.conda_prefix}));
        cmd.addArgs(&[_][]const u8{
            "-lstdc++",
            "-lz",
        });

        for (extra_libs) |lib| {
            cmd.addArg(lib);
        }

        cmd.addArgs(&[_][]const u8{ "-o", self.builder.fmt("{s}/{s}", .{ BIN_OUT, output_name }) });

        return cmd;
    }

    /// Build executable with incremental compilation
    ///
    /// Parameters:
    /// - `sources`: list of source files to compile
    /// - `common_objs`: optional list of precompiled object files to link against
    /// - `common_obj_steps`: optional list of build steps for common object files
    /// - `output`: name of the output executable
    /// - `libs`: list of libraries to link against
    /// - `flags`: list of compilation flags
    /// - `includes`: list of include directories
    ///
    /// Returns: build step for linking the executable
    pub fn build(
        self: Self,
        sources: []const []const u8,
        common_objs: ?[]const []const u8,
        common_obj_steps: ?[]const *std.Build.Step.Run,
        output: []const u8,
        libs: []const []const u8,
        flags: []const []const u8,
        includes: []const []const u8,
    ) *std.Build.Step.Run {
        var exe_objects = std.array_list.Managed([]const u8).init(self.builder.allocator);
        var exe_obj_steps = std.array_list.Managed(*std.Build.Step.Run).init(self.builder.allocator);

        // Add common objects if provided
        if (common_objs) |objs| {
            for (objs) |obj| {
                exe_objects.append(obj) catch @panic("OOM");
            }
        }

        // Compile sources with incremental checking
        for (sources) |source| {
            const obj_path = self.builder.fmt("{s}/{s}.o", .{ OBJ_OUT, self.sourceToObjectName(source) });
            exe_objects.append(obj_path) catch @panic("OOM");

            if (self.createConditionalObjectCmd(source, flags, includes)) |obj_cmd| {
                obj_cmd.step.dependOn(&self.mkdir_obj_cmd.step);
                exe_obj_steps.append(obj_cmd) catch @panic("OOM");
            }
        }

        const link_cmd = self.createLinkCmd(exe_objects.items, output, libs);
        link_cmd.step.dependOn(&self.mkdir_bin_cmd.step);

        // Depend on common object compilations if provided
        if (common_obj_steps) |steps| {
            for (steps) |obj_step| {
                link_cmd.step.dependOn(&obj_step.step);
            }
        }

        // Depend on source-specific object compilations
        for (exe_obj_steps.items) |obj_step| {
            link_cmd.step.dependOn(&obj_step.step);
        }

        return link_cmd;
    }
};

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

    const gxx_path = b.findProgram(&[_][]const u8{"g++"}, &[_][]const u8{
        b.fmt("{s}/bin", .{conda_prefix}),
        "/usr/bin",
    }) catch "g++";

    std.debug.print("Using g++ path: {s}\n", .{gxx_path});

    // Initialize Builder
    const builder = Builder.init(b, gxx_path, conda_prefix);

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

    // Compile common sources first to be used by multiple executables
    var all_object_files = std.array_list.Managed([]const u8).init(b.allocator);
    var all_object_steps = std.array_list.Managed(*std.Build.Step.Run).init(b.allocator);

    for (common_sources) |source| {
        const obj_path = b.fmt("{s}/{s}.o", .{ OBJ_OUT, builder.sourceToObjectName(source) });
        all_object_files.append(obj_path) catch @panic("OOM");

        if (builder.createConditionalObjectCmd(source, &common_flags, &common_includes)) |obj_cmd| {
            obj_cmd.step.dependOn(&builder.mkdir_obj_cmd.step);
            all_object_steps.append(obj_cmd) catch @panic("OOM");
        }
    }

    // Build executables using the Builder
    const pipeline_cmd = builder.build(
        &[_][]const u8{ "src/main.cpp", "src/hnswpq/search.cpp" },
        all_object_files.items,
        all_object_steps.items,
        "pipeline",
        &[_][]const u8{ "-lopenvino", "-lomp", "-lfaiss" },
        &common_flags,
        &common_includes,
    );

    const inference_cmd = builder.build(
        &[_][]const u8{"src/inference/test_inference.cpp"},
        all_object_files.items,
        all_object_steps.items,
        "inference",
        &[_][]const u8{"-lopenvino"},
        &common_flags,
        &common_includes,
    );

    const hnswpq_cmd = builder.build(
        &[_][]const u8{"src/hnswpq/index.cpp"},
        all_object_files.items,
        all_object_steps.items,
        "hnswpq_index",
        &[_][]const u8{ "-lopenvino", "-lomp", "-lfaiss" },
        &common_flags,
        &common_includes,
    );

    // Set up build step dependencies
    const install_step = b.getInstallStep();
    install_step.dependOn(&pipeline_cmd.step);
    install_step.dependOn(&inference_cmd.step);
    install_step.dependOn(&hnswpq_cmd.step);

    // Individual build steps
    const pipeline_step = b.step("pipeline", "Build only the pipeline executable");
    pipeline_step.dependOn(&pipeline_cmd.step);

    const inference_step = b.step("inference", "Build only the inference executable");
    inference_step.dependOn(&inference_cmd.step);

    const index_step = b.step("index", "Build only the index executable");
    index_step.dependOn(&hnswpq_cmd.step);

    // Build step for common object files
    const objects_step = b.step("objects", "Build common object files");
    for (all_object_steps.items) |obj_step| {
        objects_step.dependOn(&obj_step.step);
    }

    // Clean step to remove build artifacts
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", BIN_OUT, OBJ_OUT });
    clean_step.dependOn(&clean_cmd.step);
}
