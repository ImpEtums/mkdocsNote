## Chapter 1. Complexity Analysis

### 1. 1 iteration
1. *for* loop
2. *while* loop
3. nested loop

### 1. 2 recursion
It consists of two main stages:
1. The program calls itself deeper and deeper, usually passing smaller or simpler arguments, until it reaches a "termination condition".
2. After the "termination condition" is triggered, the program returns from the deepest recursive function level by level, aggregating the results of each level.

From an implementation perspective, recursive code mainly contains three elements:

1. Termination conditions: used to decide when to go from "recursive" to "reductive".
2. Recursive call: the function calls itself, usually with smaller or simpler arguments.
3. Return result: returning the result of the current recursion level to the previous one.

sample code:
```c 
//recursion.c
//calculate 1+2+...+n
int recur(int n){
    //teimination condition
    if (n == 1)
        return 1;
    //recursive call
    int res = recur(n - 1);
    //return result
    return n + res;
}
```
```py 
#recursion.py
def recur(n):
    if n==1:
        return 1
    res = recur(n-1)
    return n+res
```
#### 1. 2. 1 call stack


#### 1. 2. 2 tail recursion
