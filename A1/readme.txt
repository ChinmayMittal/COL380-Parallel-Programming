Number of approaches: 4

1:Idea: Naive Non Threaded, simple matrix multiplications without any optimizations.
1:Why attempted: I implemented this as a baseline to compare parallel optimizations and also to verify that my I/O operations, data structures for storing matrices are correct
1:Results: Works faster for smaller matrices (no overhead in thread creation and management), takes excessive time for larger matrices and does not scale well.
Code took 1.2 ms for smaller input file and too much time for larger input file (did not end for long time).
1:Drawback: This does not use any parallelization and does not scale on large matrices or does not make use of more cores on the machine.

2:Idea I divide the work into calculating rows of blocks of the output matrix, each block row in the output can be computed independent of the other and thus make use of parallelization.
Also multiplying the matrices by dividing them into blocks and multiplying these blocks instead of individiual element mulitplications can make much better use of cache during parallelization.
2:Why attempted I wanted to divide the matrix computations such that tasks can be completed independent of one another without the need for synchronization between threads.
2:Results: I tested with 8 cores and 8 threads, time taken for large matrix: 9141.68 ms, time for small matrix: 0.41 ms (After optimizations discussed in approach 4, also applicable for approach 2)
2:Drawback: Each task is large, hence number of tasks (granularity) is large. If number of rows in the output are small than we don't make effective use of parallelization

3:Idea Further increase the granularity of the task graph, each task now represents computing one output block.
3:Why attempted: The main idea is to increase the granularity of the task graph which can ensure there are always enough tasks to keep all the cores busy and also not very small
tasks which can be counterproductive if the overhead of scheduling tasks becomes comparable to the work done in each task.
3:Results: See section 4
3:Drawback: There is scope of further optimizations as discussed in the next sections

4:Idea: I optimized approach 3 keeping all other things (in terms of parallelization) the same. Instead of multiplying a row of blocks with a column of blocks. Using the fact that the matrix 
is symmetric I multiplied a block of rows with another block of rows, now each block block multiplication becomes multiplying the block with the transpose of the other block. This is 
particularly beneficial for cache optimizations since C++ stores elements row wise. Hence we exploit spatial locality of caches. Also each block is also stored as a 1-D array instead of 
2-D array since in dynamic 2-D vectors used in my implementation each row of the block was not stored consecutively initially. I also clear memory for output blocks which are computed to be zero.
I also made all function calls inline saving overhead for creating call stacks. I removed redundant computations, such as adding or multiplying with zero blocks.
4:Why attempted: The main idea was to remove bottlenecks in the code and optimize it keeping the parallel skeleton of the code the same. This lead to about a 10x improvement in time 
compared to the non-optimzed approach-3. I have also improved parallelization by creating number of threads same as the number of cores available.
4:Results Time for large matrix: 8437 ms and time for smaller matrix: 0.83 ms
4:Drawback My program is ineffecient memory wise because of using vector of vectors

Final Scalability analysis
--------------------------------------------------------------------------------------------------
| Non-0 input blocks    |  Non-0 output blocks  |   2 cores    |   4 cores  |  8cores | 16 cores |
--------------------------------------------------------------------------------------------------
|  2^10 (n=10000, m=25) |         5523          |   455.74     |   232.06   |  120.91 |  84.49   |
--------------------------------------------------------------------------------------------------
|  2^15 (n=100000, m=25)|        523035         |   77327      |   39602    |   20999 |  12760   |
--------------------------------------------------------------------------------------------------
|  2^20                 |                       |              |            |         |          |
--------------------------------------------------------------------------------------------------
|    2^25               |                       |              |            |         |          |
-------------------------------------------------------------------------------------------------
last inputs could not be run because of space constrains on local machine and css was down.
all tests were run on local machine because css was down.