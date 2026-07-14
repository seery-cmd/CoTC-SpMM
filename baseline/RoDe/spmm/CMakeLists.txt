aux_source_directory(./cuSparse DIR_CUSPARSE_SRCS)
aux_source_directory(./Sputnik DIR_SPUTNIK_SRCS)
aux_source_directory(./RoDe DIR_RODE_SRCS)

add_library(cuSPARSE_spmm ${DIR_CUSPARSE_SRCS})
target_link_libraries(cuSPARSE_spmm ${CUDA_cusparse_LIBRARY})

add_library(sputnik_spmm ${DIR_SPUTNIK_SRCS})
target_include_directories(sputnik_spmm PRIVATE ${UTILS_LIBS})
target_link_libraries(sputnik_spmm utils)

add_library(RoDe_spmm ${DIR_RODE_SRCS})
target_include_directories(RoDe_spmm PRIVATE ${UTILS_LIBS})
target_link_libraries(RoDe_spmm utils)