include_directories(${UTILS_LIBS})
include_directories(${Sputnik_SPMM_LIBS})
include_directories(${CUSPARSE_SPMM_LIBS})
include_directories(${RoDe_SPMM_LIBS})

add_executable(eval_spmm_f32_n128 eval_spmm_f32_n128.cu)
target_link_libraries(eval_spmm_f32_n128 utils sputnik_spmm cuSPARSE_spmm RoDe_spmm)

add_executable(eval_spmm_f32_n256 eval_spmm_f32_n256.cu)
target_link_libraries(eval_spmm_f32_n256 utils sputnik_spmm cuSPARSE_spmm RoDe_spmm)