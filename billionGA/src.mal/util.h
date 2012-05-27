#ifndef UTIL__H
#define UTIL__H

inline const char* int_to_binary(int x)
{
    int b_size = sizeof(char) * ((sizeof(int) * 8) + 1);
    char *b = (char*)malloc(b_size);
    
    b[b_size-1] = '\0';

    unsigned int starting = 1 << ((sizeof(int) * 8) - 1);
    int pos = 0;
    
    for (unsigned int z = starting; z > 0; z >>= 1)
    {        
        if ((x & z) == z) {
            b[pos] =  '1';
        } else {
            b[pos] =  '0';
        }
        
        pos++;
    }

    return b;
}

#endif
