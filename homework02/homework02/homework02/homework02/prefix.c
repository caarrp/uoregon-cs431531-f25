#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc > 2) {
        n = atoi(argv[1]); 
        seed = atoi(argv[2]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }


    // set up data 
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);  
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }


    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);

    
    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
	fprintf(stderr, "solution: %d, answer: %d", sol[err], ans[err]);
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}


void prefix_sum_p1(int* src, int* prefix, int n)
{   // hillis steele inclusive scan
	
	//fprintf(stderr, "source array:\n<");
	//for (int k = 0; k < n; k++){ fprintf(stderr,"%d, ",  src[k]);}
	//fprintf(stderr, ">\n");

	int * temp = malloc(n * sizeof(int));

	#pragma omp parallel for  
	for (int j = 0; j < n; j++){
		prefix[j] = src[j];
		temp[j] =   src[j];
	}// to copy to prefix first
	
	
	/*for (int stride = 1; stride < n; stride *= 2) {//reverse iteration
        #pragma omp parallel for
        for (int i = n - 1; i >= stride; i--) {
            prefix[i] += prefix[i - stride];
        	}
    	}*/
	
	for (int stride = 1; stride < n; stride *= 2){
		
		#pragma omp parallel for
		for (int i = 0; i < n; i++){

			if (i < stride){
				temp[i] = prefix[i];
			}else{
				prefix[i] = temp[i] + temp[i - stride];
			}
		}
		
		#pragma omp parallel for
		for (int j = 0; j < n; j++){
			temp[j] = prefix[j];
		}
		//fprintf(stderr, "After stride %d: ", stride);
        	//for (int k = 0; k < n; k++){ fprintf(stderr,"%d, ",  prefix[k]);}
        	//fprintf(stderr, "\n");
	    }	
	free(temp);	
	
	//fprintf(stderr, "prefix array:\n<");
	//for (int k = 0; k < n; k++){fprintf(stderr,"%d, ",  prefix[k]);}
	//fprintf(stderr, ">\n");
}


void prefix_sum_p2(int* src, int* prefix, int n)
{
	//fprintf(stderr, "source array:\n<");
	//for (int k = 0; k < n; k++){ fprintf(stderr,"%d, ",  src[k]);}
	//fprintf(stderr, ">\n");
	
	int *temp = malloc(n * sizeof(int));
	for (int j = 0; j < n; j++){
		temp[j] = src[j];
		prefix[j] = src[j];
	}
	
	//REDUCTION
	for (int stride = 1; stride < n; stride *= 2){
	
	#pragma omp parallel for
		for (int i = (stride*2 - 1); i < n; i+= (stride*2)){
			
			prefix[i] = temp[i] + temp[i - stride];
			temp[i] = prefix[i];
			
		}
		//fprintf(stderr, "After stride %d: ", stride);
                //for (int k = 0; k < n; k++){ fprintf(stderr,"%d, ",  prefix[k]);}
                //fprintf(stderr, "\n");
	}

	//fprintf(stderr, "prefix array:\n<");
	//for (int k = 0; k < n; k++){fprintf(stderr,"%d, ",  prefix[k]);}
	//fprintf(stderr, ">\n");
	
	int final = prefix[n-1];
	temp[n-1] = 0; prefix[n-1] = 0;

	//DOWNSWEEP
	for (int stride = n/2; stride >= 1; stride /= 2){
		
	#pragma omp parallel for
		for (int i = (stride*2 - 1); i < n; i+= stride*2 ){
			prefix[i] = temp[i] + temp[i - stride];
			prefix[i-stride] = temp[i];

			temp[i] = prefix[i];
			temp[i-stride] = prefix[i-stride];
		}
	        //fprintf(stderr, "After stride %d: ", stride);
                //for (int k = 0; k < n; k++){ fprintf(stderr,"%d, ",  prefix[k]);}
                //fprintf(stderr, "\n");
	}
	
	prefix[n-1] = final;
	// make it INCLUSIVE
	#pragma omp parallel for
	for (int k = 0; k < n-1; k++){
		prefix[k] = temp[k+1];
	}

	 
        //fprintf(stderr, "prefix array:\n<");
        //for (int k = 0; k < n; k++){fprintf(stderr,"%d, ",  prefix[k]);}
        //fprintf(stderr, ">\n");
}



