## Chapter 2. Sorting

### 2.1 Insertion Sort

```c
//insertion_sort.c
//min to max
void insertion_sort(int arr[], int len){
    for(int i=1; i<len; ++i>){
        int key = arr[i];
        int j = i-1;
        while(j>=0 && arr[j] > key){
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}
```