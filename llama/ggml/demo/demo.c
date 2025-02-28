#include "ggml.h"
#include <string.h>
#include <stdio.h>

int main(void) {
    // 创建矩阵数据
    const int rows_A = 4, cols_A = 2;
    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };
    const int rows_B = 3, cols_B = 2;
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    // 分配ggml_context 来存储 tensor data
    // 提前分配空间
    size_t ctx_size = 0;
    // 分配的存储矩阵的实际数据内存大小
    ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32);
    ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32);
    ctx_size += rows_A * rows_B * ggml_type_size(GGML_TYPE_F32);
    // 分配的是元数据内存 用于存储张量的维度、数据类型等信息
    ctx_size += 3 * ggml_tensor_overhead();
    ctx_size += ggml_graph_overhead();
    ctx_size += 1024;

    struct ggml_init_params params = {
        /*.mem_size =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc =*/ false,
    };

    struct ggml_context *ctx = ggml_init(params);

    // 创建tensor 并赋值
    struct ggml_tensor *tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows_A, cols_A);
    struct ggml_tensor *tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows_B, cols_B);
    memcpy(tensor_a->data, matrix_A, ggml_nbtyes(tensor_a));
    memcpy(tensor_b->data, matrix_B, ggml_nbtyes(tensor_a));

    // 创建计算图存储矩阵乘法
    struct ggml_cgraph *gf = ggml_new_graph(ctx);

    struct ggml_tensor *result = ggml_mul_mat(ctx, tensor_a, tensor_b);

    // 将result tensor添加到计算图中
    ggml_build_forward_expand(gf, result);
    
    
    // 执行计算图
    int n_threads = 1;
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // 输出结果
    // ne存储的是tensor的维度信息 ne[0]表示行数 ne[1]表示列数
    float * result_data = (float *)result->data;
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

    // 释放分配空间
    ggml_free(ctx);
    return 0;
}