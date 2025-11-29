const std = @import("std");

const CONDA_ENV = "DeepReadMapper";
const ZIG_OUT = "zig-out/bin";

// Function to create optimized compile command
fn createCompileCmd(builder: *std.Build, compiler_path: []const u8, sources: []const []const u8, output_name: []const u8, extra_libs: []const []const u8, conda_prefix: []const u8, common_flags: []const []const u8, common_includes: []const []const u8) *std.Build.Step.Run {
    const cmd = builder.addSystemCommand(&[_][]const u8{compiler_path});

    // Add common flags
    for (common_flags) |flag| {
        cmd.addArg(flag);
    }

    // Add common includes
    for (common_includes) |inc| {
        cmd.addArg(inc);
    }

    // Add conda include
    cmd.addArg(builder.fmt("-isystem{s}/include", .{conda_prefix}));

    // Add source files
    for (sources) |src| {
        cmd.addArg(src);
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
    cmd.addArgs(&[_][]const u8{ "-o", builder.fmt("{s}/{s}", .{ ZIG_OUT, output_name }) });

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
        std.debug.print("Current CONDA_RPEFIX: {s}\n", .{conda_prefix});
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

    // Common source files shared across executables
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
        // "-DNDEBUG", // Disable debug assertions for better performance
        // "-flto", // Link-time optimization
        // "-ffast-math", // Fast math optimizations
    };

    // Common include directories
    const common_includes = [_][]const u8{
        "-Iincludes",
        "-Iincludes/inference",
        "-Iincludes/utils",
        "-Iincludes/hnswlib_dir", // Original HNSW. Links with
        // "-Iincludes/hnswm", // Minh's HNSW
        "-Iincludes/gann_hnsw", // GANN's HNSW
        "-Iincludes/hnswpq", // FAISS's HNSW
        "-Iexternal/cereal/include",
        "-Iexternal/cnpy",
    };

    // 1. Pipeline executable
    const pipeline_sources = common_sources ++ [_][]const u8{
        "src/main.cpp",
        "src/hnswpq/search.cpp",
    };

    const compile_cmd = createCompileCmd(b, gcc_path, &pipeline_sources, "pipeline", &[_][]const u8{
        "-lopenvino",
        "-lomp",
        "-lfaiss",
    }, conda_prefix, &common_flags, &common_includes);

    // Ensure output directory exists
    const mkdir_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", ZIG_OUT });
    compile_cmd.step.dependOn(&mkdir_cmd.step);

    // 2. Inference test executable
    const inference_sources = common_sources ++ [_][]const u8{
        "src/inference/test_inference.cpp",
    };

    const inference_cmd = createCompileCmd(b, gcc_path, &inference_sources, "inference", &[_][]const u8{
        "-lopenvino",
    }, conda_prefix, &common_flags, &common_includes);
    inference_cmd.step.dependOn(&mkdir_cmd.step);

    // 3. HNSWPQ Index executable
    const hnswpq_index_sources = common_sources ++ [_][]const u8{
        "src/hnswpq/index.cpp",
    };

    const hnswpq_cmd = createCompileCmd(b, gcc_path, &hnswpq_index_sources, "hnswpq_index", &[_][]const u8{
        "-lopenvino",
        "-lomp",
        "-lfaiss",
    }, conda_prefix, &common_flags, &common_includes);
    hnswpq_cmd.step.dependOn(&mkdir_cmd.step);

    // Enable parallel compilation by NOT making executables depend on each other
    const install_step = b.getInstallStep();
    install_step.dependOn(&compile_cmd.step);
    install_step.dependOn(&inference_cmd.step);
    install_step.dependOn(&hnswpq_cmd.step);

    // Individual build steps for faster development iteration
    const pipeline_step = b.step("pipeline", "Build only the pipeline executable");
    pipeline_step.dependOn(&compile_cmd.step);

    const inference_step = b.step("inference", "Build only the inference executable");
    inference_step.dependOn(&inference_cmd.step);

    const index_step = b.step("index", "Build only the index executable");
    index_step.dependOn(&hnswpq_cmd.step);

    // Fast debug build (no optimization, faster compilation)
    const debug_step = b.step("debug", "Fast debug build with minimal optimization");
    const debug_flags = [_][]const u8{
        "-std=c++17",
        "-g", // Debug symbols
        "-O0", // No optimization
        "-fopenmp",
        "-Wall",
    };

    const debug_cmd = b.addSystemCommand(&[_][]const u8{gcc_path});
    for (debug_flags) |flag| {
        debug_cmd.addArg(flag);
    }
    for (common_includes) |inc| {
        debug_cmd.addArg(inc);
    }
    debug_cmd.addArg(b.fmt("-isystem{s}/include", .{conda_prefix}));
    for (pipeline_sources) |src| {
        debug_cmd.addArg(src);
    }
    debug_cmd.addArg(b.fmt("-L{s}/lib", .{conda_prefix}));
    debug_cmd.addArgs(&[_][]const u8{
        "-lopenvino",
        "-lomp",
        "-lfaiss",
        "-lstdc++",
        "-lz",
        "-o",
        b.fmt("{s}/pipeline_debug", .{ZIG_OUT}),
    });
    debug_cmd.step.dependOn(&mkdir_cmd.step);
    debug_step.dependOn(&debug_cmd.step);

    // Run step for convenience
    const run_cmd = b.addSystemCommand(&[_][]const u8{b.fmt("./{s}/pipeline", .{ZIG_OUT})});
    run_cmd.step.dependOn(&compile_cmd.step);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the pipeline");
    run_step.dependOn(&run_cmd.step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", ZIG_OUT });
    clean_step.dependOn(&clean_cmd.step);
}
