#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA  
#include "ggml-cuda.h"
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS_A 4
#define COLS_A 2

#define ROWS_B 3
#define COLS_B 2

int main(void) {
    // 创建矩阵数据
    float matrix_A[ROWS_A * COLS_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    float matrix_B[ROWS_B * COLS_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    // 1.初始化后端
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s:Using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0);
    if (backend == NULL) {
        fprintf(stderr, "%s:Failed to initialize CUDA backend\n", __func__);
    }
#endif

    // 如果没有gpu后端启动，初始化CPU后端
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }


    // 2.分配ggml_context 存储tensor信息
    // 提前计算空间进行分配
    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead();
    // 不要分配其他空间？

    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = true, // 需要自己进行mem_buffer的分配
    };
    
    struct ggml_context *ctx = ggml_init(params);
    
    // 3.创建tensor 元数据(shape 以及type)
    struct ggml_tensor *tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, COLS_A, ROWS_A);
    struct ggml_tensor *tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, COLS_B, ROWS_B);


    // 4.通过backend_buffer来存储tensor (分配buffer 因为前面no_alloc=true)
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // 5.将tensor data从RAM拷贝到backend_buffer
    ggml_backend_tensor_set(tensor_a, matrix_A, 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, matrix_B, 0, ggml_nbytes(tensor_b));

    // 6.创建计算图
    struct ggml_cgraph *gf = NULL;
    // 创建临时context构建计算图
    struct ggml_context *ctx_cgrapa = NULL;
    {
        struct ggml_init_params params0 = {
            .mem_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            .mem_buffer = NULL,
            .no_alloc = true,
        };
        ctx_cgrapa = ggml_init(params0);
        gf = ggml_new_graph(ctx_cgrapa);

        // result tensor
        struct ggml_tensor *result0 = ggml_mul_mat(ctx_cgrapa, tensor_a, tensor_b);
        ggml_build_forward_expand(gf, result0);
    }
    // 7.将tensor添加到计算图 (no_alloc=true, 需要自己分配内存)
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_add(gallocr, gf);

    // 8.使用后端调度计算图(跳过)

    // 9.执行计算图
    int  n_threads = 1;
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    ggml_backend_graph_compute(backend, gf);

    // 10.输出结果
    // output在gf计算图的最后一个
    struct ggml_tensor *result = gf->nodes[gf->n_nodes - 1];
    float *result_data = malloc(ggml_nbytes(result));
    // result_tensor 存储在后端buffer中，需要拷贝到RAM中
    ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
    prinft("mul mat (%d x %d) (transposed result):\n", (int)result->ne[0], (int)result->ne[1]);
    for (int j = 0; j < result->ne[1]; j++) {
        if (j > 0) {
            printf("\n");
        }
        for (int i = 0; i < result->ne[0]; i++) {
            printf("%f ", result_data[j * result->ne[0] + i]);
        }   
    }
    printf("]\n");
    free(result_data);

    // 11.释放内存
    ggml_free(ctx);
    ggml_free(ctx_cgrapa);
    ggml_backend_free_buffer(buffer);
    ggml_backend_free(backend);
    ggml_gallocr_free(gallocr);
    return 0;   

}