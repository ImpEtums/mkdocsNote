
## Memory-Management-in-C

 | Functions | Desceriptions |
 | :---: | :---: |
 | void *calloc(int num, int size); | This function allocates an array of **num** elements each of which size in bytes will be **size**. It results in the allocation of **num * size** bytes of memory, and each byte has a value of zero. |
 | void *malloc(int num); | This function allocates an array of **num** bytes and leave them uninitialized. |
 | void *realloc(void *address, int newsize); | This function re-allocates memory extending it upto newsize. |
 | void free(void *address); | This function frees the memory block pointed to by **address**, freeing dynamically allocated memory space.|

 