#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

int sum_bits_from_int(int data) {
    unsigned int sum = 0;
    unsigned int starting = 1 << ((sizeof(int) * 8)-1);
    int shifts = 32;
    
    for (unsigned int z = starting; z > 0; z >>= 1)
    {
        shifts--;
        
        fprintf(stdout, "z = %u\n", z);
        fprintf(stdout, "data & z = %u\n", data & z);
        sum += (data & z) >> shifts;
        fprintf(stdout, "(data & z) >> shifts = %u\n", (data & z) >> shifts);
    }
    
    return sum;   
}

const char* int_to_binary(int x)
{
    int b_size = sizeof(char) * ((sizeof(int) * 8) + 1);
    char *b = (char*)malloc(b_size);
    
    b[b_size-1] = '\0';

    int pos = 0;
    unsigned int starting = 1 << ((sizeof(int) * 8)-1);
    //fprintf(stdout, "z(-1) = %d\n", starting);
    
    for (unsigned  int z = starting; z > 0; z >>= 1)
    {        
        //fprintf(stdout, "z = %d\n", z);
        
        if ((x & z) == z) {
            b[pos] =  '1';
        } else {
            b[pos] =  '0';
        }
        
        pos++;
    }

    return b;
}

int main(int argc, char **argv) {
    char current_block_sample[4];
    
    char aux, offset, block;
    
    for (int tid = 0; tid < 32; tid++) {
        block = tid >> 3;
        offset = tid & ((1 << 3)-1);
        
        fprintf(stdout, "tid %d block %d offset %d", tid, block, offset);
        
        if (tid % 2 == 1) {
            // 1
            current_block_sample[block] = current_block_sample[block] | (1 << offset);
        } else {
            // 0
            current_block_sample[block] = current_block_sample[block] & ~(1 << offset);
        }
        
        fprintf(stdout, "\n");
    }
    
    for (int i = 0; i < 1; i++) {
        printf("%s (%d)\n", int_to_binary(((int*)current_block_sample)[0]), ((int*)current_block_sample)[0]);
        printf("sum %d\n", sum_bits_from_int(((int*)current_block_sample)[0]));
    }
}
