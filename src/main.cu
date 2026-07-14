
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include "readMtx/utils.h"
#include "include/compare_cu.cuh"

/**
 * author:seery_hnu_dq
 * data:2025.4.17
 * -----have a good ACG day------
 */
using namespace std;

int main(int argc,char *argv[])
{
    //int length_cache = 6 * 1024;
    printf("input matrix = %s\ntest matrix = %s\n",argv[1],argv[2]);
    main_SpVV_T_compare(argv[1]);
    printf("main done!\n");

}
