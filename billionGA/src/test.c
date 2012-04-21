#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

const char* byte_to_binary(int x)
{
    char *b;
    b = (char*)malloc(sizeof(char) * 9);
    b[8] = '\0';

    int pos = 0;
    for (int z = (1<<7); z > 0; z >>= 1)
    {        
        if ((x & z) == z) {
            fprintf(stdout, "z = %d (1)\n", z);
            b[pos] =  '1';
        } else {
            fprintf(stdout, "z = %d (0)\n", z);
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
    
    for (int i = 0; i < 4; i++) {
        printf("%s (%d)\n", byte_to_binary(current_block_sample[0]), current_block_sample[i]);
    }
}
