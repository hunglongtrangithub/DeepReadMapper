const std = @import("std");

const CONDA_ENV = "DeepReadMapper";
const ZIG_OUT = "zig-out/bin";

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

    // Build all sources into one executable using g++ directly
    const all_sources = [_][]const u8{
        "src/main.cpp",
        "src/hnswpq/search.cpp",
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

    // Find g++ (prefer conda's if available)
    const gcc_path = b.findProgram(&[_][]const u8{"g++"}, &[_][]const u8{
        b.fmt("{s}/bin", .{conda_prefix}),
        "/usr/bin", // On Ubuntu Linux
    }) catch "g++";

    std.debug.print("Using g++ path: {s}\n", .{gcc_path});

    // Create the g++ command with explicit path
    const compile_cmd = b.addSystemCommand(&[_][]const u8{gcc_path});

    // Add compiler flags
    compile_cmd.addArgs(&[_][]const u8{
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-march=native",
        "-Wall",
    });

    // Add include directories
    compile_cmd.addArgs(&[_][]const u8{
        "-Iincludes",
        "-Iincludes/inference",
        "-Iincludes/utils",
        "-Iincludes/hnswlib_dir", // Original HNSW. Links with
        // "-Iincludes/hnswm", // Minh's HNSW
        // "-Iincludes/gann_hnsw", // GANN's HNSW
        "-Iincludes/hnswpq", // FAISS's HNSW
        "-Iexternal/cereal/include",
        "-Iexternal/cnpy",
    });

    // Add conda include (use -isystem for system headers)
    compile_cmd.addArg(b.fmt("-isystem{s}/include", .{conda_prefix}));

    // Add all source files
    for (all_sources) |src| {
        compile_cmd.addArg(src);
    }

    // Add library paths
    compile_cmd.addArg(b.fmt("-L{s}/lib", .{conda_prefix}));

    // Link libraries
    compile_cmd.addArgs(&[_][]const u8{
        "-lopenvino",
        "-lomp",
        "-lfaiss",
        "-lz",
        "-lstdc++",
    });

    // Output binary
    compile_cmd.addArgs(&[_][]const u8{ "-o", b.fmt("{s}/pipeline", .{ZIG_OUT}) });

    // Ensure output directory exists
    const mkdir_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", ZIG_OUT });
    compile_cmd.step.dependOn(&mkdir_cmd.step);

    // Install step
    const install_step = b.getInstallStep();
    install_step.dependOn(&compile_cmd.step);

    // 2. Inference test executable
    const inference_sources = [_][]const u8{
        "src/inference/test_inference.cpp",
        "external/cnpy/cnpy.cpp",
        "src/inference/vectorize.cpp",
        "src/inference/fast_model.cpp",
        "src/inference/preprocess.cpp",
        "src/utils/parse_inputs.cpp",
        "src/utils/utils.cpp",
        "src/utils/tok2index.cpp",
        "src/utils/post_processor.cpp",
        "src/utils/metrics.cpp",
        "src/utils/reranker.cpp",
    };

    const inference_cmd = b.addSystemCommand(&[_][]const u8{gcc_path});
    inference_cmd.addArgs(&[_][]const u8{
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-march=native",
        "-Wall",
        "-Iincludes",
        "-Iincludes/inference",
        "-Iincludes/utils",
        "-Iincludes/hnswlib_dir",
        "-Iincludes/hnswpq",
        "-Iexternal/hnswlib",
        "-Iexternal/cnpy",
    });
    inference_cmd.addArg(b.fmt("-isystem{s}/include", .{conda_prefix}));
    for (inference_sources) |src| {
        inference_cmd.addArg(src);
    }
    inference_cmd.addArg(b.fmt("-L{s}/lib", .{conda_prefix}));
    inference_cmd.addArgs(&[_][]const u8{
        "-lopenvino",
        "-lz",
        "-lstdc++",
        "-o",
        b.fmt("{s}/inference", .{ZIG_OUT}),
    });
    inference_cmd.step.dependOn(&mkdir_cmd.step);
    install_step.dependOn(&inference_cmd.step);

    // 3. HNSWPQ Index executable
    const hnswpq_sources = [_][]const u8{
        "src/hnswpq/index.cpp",
        "external/cnpy/cnpy.cpp",
        "src/inference/vectorize.cpp",
        "src/inference/fast_model.cpp",
        "src/inference/preprocess.cpp",
        "src/utils/parse_inputs.cpp",
        "src/utils/utils.cpp",
        "src/utils/tok2index.cpp",
        "src/utils/post_processor.cpp",
        "src/utils/metrics.cpp",
        "src/utils/reranker.cpp",
    };

    const hnswpq_cmd = b.addSystemCommand(&[_][]const u8{gcc_path});
    hnswpq_cmd.addArgs(&[_][]const u8{
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-march=native",
        "-Wall",
        "-Iincludes",
        "-Iincludes/inference",
        "-Iincludes/utils",
        "-Iincludes/hnswlib_dir",
        "-Iincludes/hnswpq",
        "-Iexternal/hnswlib",
        "-Iexternal/cnpy",
    });
    hnswpq_cmd.addArg(b.fmt("-isystem{s}/include", .{conda_prefix}));
    for (hnswpq_sources) |src| {
        hnswpq_cmd.addArg(src);
    }
    hnswpq_cmd.addArg(b.fmt("-L{s}/lib", .{conda_prefix}));
    hnswpq_cmd.addArgs(&[_][]const u8{
        "-lopenvino",
        "-lomp",
        "-lfaiss",
        "-lz",
        "-lstdc++",
        "-o",
        b.fmt("{s}/hnswpq_index", .{ZIG_OUT}),
    });
    hnswpq_cmd.step.dependOn(&mkdir_cmd.step);
    install_step.dependOn(&hnswpq_cmd.step);

    // Run step for convenience
    const run_cmd = b.addSystemCommand(&[_][]const u8{b.fmt("./{s}/pipeline", .{ZIG_OUT})});
    run_cmd.step.dependOn(&compile_cmd.step);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the pipeline");
    run_step.dependOn(&run_cmd.step);
}
