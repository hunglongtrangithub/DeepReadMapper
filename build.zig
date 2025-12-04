const std = @import("std");

const CONDA_ENV = "DeepReadMapper";
const ZIG_OUT = "zig-out";
const BIN_OUT = ZIG_OUT ++ "/bin"; // Binaries output directory
const OBJ_OUT = ZIG_OUT ++ "/obj"; // Object files directory

/// Function to compile a single source to object file
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

    // Output object file
    const obj_name = std.fs.path.stem(std.fs.path.basename(source));
    cmd.addArgs(&[_][]const u8{ "-o", builder.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name }) });

    return cmd;
}

/// Function to ONLY link pre-compiled object files (no compilation)
fn createLinkCmd(builder: *std.Build, compiler_path: []const u8, all_object_files: []const []const u8, output_name: []const u8, extra_libs: []const []const u8, conda_prefix: []const u8) *std.Build.Step.Run {
    const cmd = builder.addSystemCommand(&[_][]const u8{compiler_path});

    // Only linking flags (no optimization flags needed for linking)
    cmd.addArgs(&[_][]const u8{
        "-fopenmp", // Still needed for linking OpenMP
    });

    // Add ALL pre-compiled object files (common + main sources)
    for (all_object_files) |obj| {
        cmd.addArg(obj);
    }

    // Add library path and common libraries
    cmd.addArg(builder.fmt("-L{s}/lib", .{conda_prefix}));
    cmd.addArgs(&[_][]const u8{
        "-lstdc++",
        "-lz",
    });

    // Add extra libraries
    for (extra_libs) |lib| {
        cmd.addArg(lib);
    }

    // Output binary
    cmd.addArgs(&[_][]const u8{ "-o", builder.fmt("{s}/{s}", .{ BIN_OUT, output_name }) });

    return cmd;
}

pub fn build(b: *std.Build) void {
    // Get conda prefix for finding libraries
    const conda_prefix = std.posix.getenv("CONDA_PREFIX") orelse {
        std.debug.print("Error: CONDA_PREFIX not set. Please activate conda environment:\n", .{});
        std.debug.print("  conda activate {s}\n", .{CONDA_ENV});
        std.process.exit(1);
    };

    // Verify it's the right environment
    if (std.mem.indexOf(u8, conda_prefix, CONDA_ENV) == null) {
        std.debug.print("Current CONDA_PREFIX: {s}\n", .{conda_prefix});
        std.debug.print("Please run: conda activate {s}\n", .{CONDA_ENV});
        std.process.exit(1);
    }

    std.debug.print("Using conda prefix: {s}\n", .{conda_prefix});

    // Find g++ (prefer conda's if available)
    const gcc_path = b.findProgram(&[_][]const u8{"g++"}, &[_][]const u8{
        b.fmt("{s}/bin", .{conda_prefix}),
        "/usr/bin", // On Ubuntu Linux
    }) catch "g++";

    std.debug.print("Using g++ path: {s}\n", .{gcc_path});

    // Common source files that will be compiled to object files
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

    // Common compiler flags
    const common_flags = [_][]const u8{
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-march=native",
        "-Wall",
    };

    // Common include directories
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

    // Ensure output directories exist
    const mkdir_bin_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", BIN_OUT });
    const mkdir_obj_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", OBJ_OUT });

    // Compile ALL sources to object files (common + main sources)
    var all_object_files = std.array_list.Managed([]const u8).init(b.allocator);
    var all_object_steps = std.array_list.Managed(*std.Build.Step.Run).init(b.allocator);

    // Compile common sources to object files
    for (common_sources) |source| {
        const obj_cmd = createObjectCmd(b, gcc_path, source, conda_prefix, &common_flags, &common_includes);
        obj_cmd.step.dependOn(&mkdir_obj_cmd.step);

        const obj_name = std.fs.path.stem(std.fs.path.basename(source));
        const obj_path = b.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name });
        all_object_files.append(obj_path) catch @panic("OOM");
        all_object_steps.append(obj_cmd) catch @panic("OOM");
    }

    // 1. Pipeline executable - compile main sources to objects first
    const pipeline_main_sources = [_][]const u8{
        "src/main.cpp",
        "src/hnswpq/search.cpp",
    };

    var pipeline_objects = std.array_list.Managed([]const u8).init(b.allocator);
    var pipeline_obj_steps = std.array_list.Managed(*std.Build.Step.Run).init(b.allocator);

    // Add common object files
    for (all_object_files.items) |obj| {
        pipeline_objects.append(obj) catch @panic("OOM");
    }

    // Compile pipeline-specific sources to objects
    for (pipeline_main_sources) |source| {
        const obj_cmd = createObjectCmd(b, gcc_path, source, conda_prefix, &common_flags, &common_includes);
        obj_cmd.step.dependOn(&mkdir_obj_cmd.step);

        const obj_name = std.fs.path.stem(std.fs.path.basename(source));
        const obj_path = b.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name });
        pipeline_objects.append(obj_path) catch @panic("OOM");
        pipeline_obj_steps.append(obj_cmd) catch @panic("OOM");
    }

    // Link pipeline executable using ONLY object files
    const pipeline_cmd = createLinkCmd(b, gcc_path, pipeline_objects.items, "pipeline", &[_][]const u8{
        "-lopenvino",
        "-lomp",
        "-lfaiss",
    }, conda_prefix);

    pipeline_cmd.step.dependOn(&mkdir_bin_cmd.step);
    // Depend on ALL object compilations (common + pipeline-specific)
    for (all_object_steps.items) |obj_step| {
        pipeline_cmd.step.dependOn(&obj_step.step);
    }
    for (pipeline_obj_steps.items) |obj_step| {
        pipeline_cmd.step.dependOn(&obj_step.step);
    }

    // 2. Inference executable - similar pattern
    const inference_main_sources = [_][]const u8{
        "src/inference/test_inference.cpp",
    };

    var inference_objects = std.array_list.Managed([]const u8).init(b.allocator);
    var inference_obj_steps = std.array_list.Managed(*std.Build.Step.Run).init(b.allocator);

    // Add common object files
    for (all_object_files.items) |obj| {
        inference_objects.append(obj) catch @panic("OOM");
    }

    // Compile inference-specific sources to objects
    for (inference_main_sources) |source| {
        const obj_cmd = createObjectCmd(b, gcc_path, source, conda_prefix, &common_flags, &common_includes);
        obj_cmd.step.dependOn(&mkdir_obj_cmd.step);

        const obj_name = std.fs.path.stem(std.fs.path.basename(source));
        const obj_path = b.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name });
        inference_objects.append(obj_path) catch @panic("OOM");
        inference_obj_steps.append(obj_cmd) catch @panic("OOM");
    }

    const inference_cmd = createLinkCmd(b, gcc_path, inference_objects.items, "inference", &[_][]const u8{
        "-lopenvino",
    }, conda_prefix);

    inference_cmd.step.dependOn(&mkdir_bin_cmd.step);
    for (all_object_steps.items) |obj_step| {
        inference_cmd.step.dependOn(&obj_step.step);
    }
    for (inference_obj_steps.items) |obj_step| {
        inference_cmd.step.dependOn(&obj_step.step);
    }

    // 3. HNSWPQ Index executable - similar pattern
    const hnswpq_main_sources = [_][]const u8{
        "src/hnswpq/index.cpp",
    };

    var hnswpq_objects = std.array_list.Managed([]const u8).init(b.allocator);
    var hnswpq_obj_steps = std.array_list.Managed(*std.Build.Step.Run).init(b.allocator);

    // Add common object files
    for (all_object_files.items) |obj| {
        hnswpq_objects.append(obj) catch @panic("OOM");
    }

    // Compile hnswpq-specific sources to objects
    for (hnswpq_main_sources) |source| {
        const obj_cmd = createObjectCmd(b, gcc_path, source, conda_prefix, &common_flags, &common_includes);
        obj_cmd.step.dependOn(&mkdir_obj_cmd.step);

        const obj_name = std.fs.path.stem(std.fs.path.basename(source));
        const obj_path = b.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name });
        hnswpq_objects.append(obj_path) catch @panic("OOM");
        hnswpq_obj_steps.append(obj_cmd) catch @panic("OOM");
    }

    const hnswpq_cmd = createLinkCmd(b, gcc_path, hnswpq_objects.items, "hnswpq_index", &[_][]const u8{
        "-lopenvino",
        "-lomp",
        "-lfaiss",
    }, conda_prefix);

    hnswpq_cmd.step.dependOn(&mkdir_bin_cmd.step);
    for (all_object_steps.items) |obj_step| {
        hnswpq_cmd.step.dependOn(&obj_step.step);
    }
    for (hnswpq_obj_steps.items) |obj_step| {
        hnswpq_cmd.step.dependOn(&obj_step.step);
    }

    // Main install step - builds all executables
    const install_step = b.getInstallStep();
    install_step.dependOn(&pipeline_cmd.step);
    install_step.dependOn(&inference_cmd.step);
    install_step.dependOn(&hnswpq_cmd.step);

    // Individual build steps for faster development iteration
    const pipeline_step = b.step("pipeline", "Build only the pipeline executable");
    pipeline_step.dependOn(&pipeline_cmd.step);

    const inference_step = b.step("inference", "Build only the inference executable");
    inference_step.dependOn(&inference_cmd.step);

    const index_step = b.step("index", "Build only the index executable");
    index_step.dependOn(&hnswpq_cmd.step);

    // Separate step to build just the common object files
    const objects_step = b.step("objects", "Build common object files");
    for (all_object_steps.items) |obj_step| {
        objects_step.dependOn(&obj_step.step);
    }

    // Clean step - now also cleans object files
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", BIN_OUT, OBJ_OUT });
    clean_step.dependOn(&clean_cmd.step);
}
