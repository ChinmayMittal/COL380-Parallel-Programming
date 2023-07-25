#include<iostream>
#include<fstream>
#include<vector>
#include<cassert>
#include <chrono>

#define PCOUT if(print) cout

using namespace std;

// from input files
int input_n, input_m, input_blocks_per_row ; // according to input
int block_size = 32 ; // my choice
int blocks_per_row ; // my choice


__host__
void read_input(const char* filename, uint16_t ** &input_arr)
{
    int n, m, k ;
    bool print = true  ;
    // filehandler opened in the input binary mode
    std::ifstream rf(filename, std::ios::in | std::ios::binary); 

    rf.read((char *)&n, sizeof(n)) ;
    rf.read((char *)&m, sizeof(m)) ; 
    rf.read((char *)&k, sizeof(k)) ; 
    
    input_n = n, input_m = m, input_blocks_per_row = n/m  ;
    
    // I redefine the block size
    blocks_per_row = (n + block_size - 1) / block_size ; // extra block if division is not perfect  


    // allocate memory on the CPU, according to my definition of block size
    input_arr = (uint16_t **) malloc(blocks_per_row * blocks_per_row * sizeof(uint16_t*));

    // initialize 2D pointer grid
    for(int idx = 0 ; idx < blocks_per_row * blocks_per_row ; idx ++ ) input_arr[idx] = NULL ; 

    PCOUT << "Input: " << filename << " " << n << " " << m << " " << k << std::endl ; 

    int blocks_read = 0, i, j ;
    unsigned char input_buffer[2] ; // buffer to read 2 bytes per matrix element
    while(blocks_read < k)
    {
        // 4 bytes each for both the indices of the blocks
        rf.read((char *)&i, sizeof(i)) ; 
        rf.read((char *)&j, sizeof(j)) ; 
        // PCOUT << "Block Idx " << i << " " << j << std::endl ; 

        int top_r = i * input_m, top_c = j * input_m ; // index of top left element of the block inside the entire array
        int block_r = top_r/block_size, block_c = top_c/block_size ; // my block indices

        // allocate block if already not allocated 
        if(input_arr[block_r*blocks_per_row + block_c] == NULL){
            input_arr[block_r*blocks_per_row + block_c] = (uint16_t *)malloc(block_size * block_size * sizeof(uint16_t));
            for(int idx = 0 ; idx < block_size * block_size ; idx ++ ) input_arr[block_r*blocks_per_row + block_c][idx] = 0 ; // initialize the block
        }

        // read the block 
        for(int blockrow = 0 ; blockrow < m ; blockrow ++ )
        {
            for(int blockcol =  0 ; blockcol < m ; blockcol ++ )
            {
                rf.read((char *)input_buffer, 2) ; // each input is two bytes
                uint16_t value = (input_buffer[1] << 8 ) | input_buffer[0] ;
                int ele_row = i * input_m + blockrow, ele_col = j * input_m + blockcol ; // index of element in entire array
                int my_block_row = ele_row % block_size, my_block_col = ele_col % block_size ; // index of element in my block
                // pointer to block and location within linearized blocks
                input_arr[block_r*blocks_per_row + block_c][my_block_row * block_size + my_block_col] = value ;  // store block in row major order
                // PCOUT << value << " " ;
            }
        }
        // PCOUT << endl ; 
        blocks_read ++ ;
    }
    rf.close() ;
    return ;
}

__host__
void print_mat(int** input_arr)
{
    for(int row = 0 ; row < input_n ; row ++ )
    {
        for(int col = 0 ; col < input_n ; col ++ )
        {
            int block_x = row/block_size, block_y = col/block_size, block_offset_x = row % block_size, block_offset_y = col % block_size ; 
            bool all_zeros = (input_arr[block_x*blocks_per_row + block_y] == NULL) ; 
            cout << (all_zeros ? 0 :input_arr[block_x*blocks_per_row + block_y][block_offset_x*block_size + block_offset_y] ) << " ";

        }
        cout << endl ;
    }
    return ;
}

__host__
void print_mat_2(uint32_t* c)
{
    for(int row = 0 ; row < input_n ; row ++ )
    {
        for(int col = 0 ; col < input_n ; col ++ )
        {
            cout << c[row*input_n + col] << " " ;
        }
        cout << "\n" ;
    }
}


__host__
int check_non_zero_blocks(uint32_t *c, bool* non_zero_output, int num_rows, int num_cols)
{
    // C => num_rows * num_cols
    int cnt = 0 ;
    for(int idx = 0 ; idx < input_blocks_per_row * input_blocks_per_row ; idx ++ ) if(non_zero_output[idx]==1) cnt ++ ;
    return cnt ;
}

__host__
void write_output(const char* filename, uint32_t* &c, bool* non_zero_output, int num_rows, int num_cols)
{
    bool print = true ;
    int non_zero_count = check_non_zero_blocks(c, non_zero_output, num_rows, num_cols) ;
    PCOUT << non_zero_count << endl ;
    std::ofstream wf ; 
    wf.open(filename, std::ios::out | std::ios::binary ) ;

    wf.write((const char *)&input_n, sizeof(int)) ;
    wf.write((const char *)&input_m, sizeof(int)) ;
    wf.write((const char *)&non_zero_count, sizeof(int)) ;
    int cnt = 12 ; 
    for(int blockrow = 0 ; blockrow < input_n / input_m ; blockrow ++ )
    {
        for( int blockcol = 0 ; blockcol < input_n / input_m ; blockcol ++ )
        {
            
            // write non zero block to the file
            if(non_zero_output[blockrow * input_blocks_per_row + blockcol])
            {
                // write index of block
                cnt += sizeof(int)*2 ; 
                wf.write((const char* )&blockrow, sizeof(int)) ;
                wf.write((const char* )&blockcol, sizeof(int)) ;

                // write block elements
                for(int i = 0 ; i < input_m ; i ++ )
                {
                    for(int j = 0 ; j < input_m ; j ++ )
                    {
                        int element_row = blockrow * input_m + i, element_col = blockcol* input_m + j ;
                        uint32_t value = c[element_row * num_cols + element_col] ;
                        wf.write((const char*)&value, sizeof(uint32_t));
                        cnt += sizeof(uint32_t) ;
                    }
                }
            }
        }
    }

    cout << cnt << endl ; 

    wf.flush() ;
    wf.close() ;
}


__global__
void sparse_matrix_mult_kernel(uint16_t**a, uint16_t**b, uint32_t*c, bool* non_zero_output, int n, int m)
{
    
    uint32_t MAX_INT = (((1ll) << 32)-1);
    int block_size = blockDim.x ;
    int blks_per_row =(n + block_size - 1)/ block_size ; 

    // shared tiled caches
    __shared__ uint32_t A_tile[32*32] ;
    __shared__ uint32_t B_tile[32*32] ;

    // output element for which this threads is responsible
    int output_row = blockIdx.y * blockDim.y + threadIdx.y ;
    int output_col = blockIdx.x * blockDim.x + threadIdx.x ;

    // initialize
    if( output_row < n && output_col < n && output_row % m == 0 && output_col % m == 0 )
    {
       non_zero_output[(output_row/m)*(n/m) + (output_col/m)] = false ;
    }

    int num_tiles = blks_per_row;
    long long int partial_product = 0 ; // will be stored to C eventually

    
    for(int tile_idx = 0 ; tile_idx < num_tiles ; tile_idx ++ )
    {
        // two tiles are (blockIdx.y, tile_idx) and (tile_idx, blockIdx.x)
        if(a[ blockIdx.y*blks_per_row + tile_idx] == NULL || b[tile_idx* blks_per_row + blockIdx.x] == NULL )
            continue ;

        // load a tile
        A_tile[threadIdx.y * blockDim.x + threadIdx.x] = a[blockIdx.y*blks_per_row + tile_idx][threadIdx.y * blockDim.x + threadIdx.x] ;
        // load b tile
        B_tile[threadIdx.y * blockDim.x + threadIdx.x] = b[tile_idx* blks_per_row + blockIdx.x][threadIdx.y * blockDim.x + threadIdx.x] ;


        // wait for all threads to load their caches
        __syncthreads() ;

        int var1 = threadIdx.y * block_size ; 
        int var2 = threadIdx.x ;
        for(int inner = 0 ; inner < block_size ; inner ++ )
        {
            // partial_product += (A_tile[ threadIdx.y * block_size + inner]) * B_tile[ inner * block_size + threadIdx.x];
            partial_product += (A_tile[var1]) * B_tile[ var2];
            var1 += 1 ;
            var2 += block_size ;
        }

        // wait for all threads to finish their computation
        __syncthreads();
    }

    if(partial_product > MAX_INT) partial_product = MAX_INT ; // represents max value, will be written by CPU

    if(output_row < n && output_col < n)
        c[output_row * n + output_col] = partial_product ;

    __syncthreads() ; 

    if( output_row < n && output_col < n && output_row % m == 0 && output_col % m == 0 )
    {
        // thread represents the top most element of output block
        // check if output block is non zero
        bool non_zero_found = false ;
        for(int row_check = output_row ; row_check < output_row + m ; row_check ++ )
        {
            for(int col_check = output_col ; col_check < output_col + m ; col_check ++ )
            {
                if(c[ row_check * n + col_check] != 0 )
                {
                    non_zero_found = true ;
                    break ; 
                }
            }
        }

        if(non_zero_found)
            // non zero bloc
            non_zero_output[(output_row/m)*(n/m) + (output_col/m)] = 1 ; 

    }
}

__host__
void read_and_send_to_gpu(const char* filename, uint16_t ** &gpu_pointer, cudaStream_t &stream)
{
    uint16_t **arr ; // 2D grid of pointers to blocks stored in a single array, with NULL pointers
    read_input(filename, arr) ; 
    
    cudaMalloc(&gpu_pointer, blocks_per_row * blocks_per_row * sizeof(uint16_t *));

    // will store pointers to GPU blocks which will be sent to the GPU
    uint16_t ** temp = (uint16_t**) malloc(input_blocks_per_row * input_blocks_per_row * sizeof(uint16_t*)) ;


     for(int idx = 0 ; idx < blocks_per_row * blocks_per_row ; idx ++ )
    {
        if(arr[idx] != NULL )
        {
            // allocate block on GPU
            cudaMalloc(&temp[idx], block_size * block_size * sizeof(uint16_t)) ;
            // send block to GPU
            cudaMemcpyAsync(temp[idx], arr[idx], block_size * block_size * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) ;
        }else{
            temp[idx] = NULL ;
        }
    }

    // copy pointer to blocks on GPU to the GPU
    cudaMemcpyAsync(gpu_pointer, temp, blocks_per_row * blocks_per_row * sizeof(uint16_t *), cudaMemcpyHostToDevice, stream) ;
    
    // clear memory on the CPU
    free(temp);

    // print_mat(arr); cout << endl ; 
    
    // free cpu memory
    for(int idx = 0 ; idx < blocks_per_row * blocks_per_row ; idx ++ ) if(arr[idx] != NULL ) free(arr[idx]) ;
    free(arr) ;

    return ;
}


__host__ 
int main(int argc, char* argv[])
{
    // A * B = C
    auto start_read = std::chrono::high_resolution_clock::now();
    uint16_t **a_dev, **b_dev ; // these pointers are for data on the GPU
   
    // two streams created for parallel asynchronous data transfer
    cudaStream_t stream1, stream2 ;
    cudaStreamCreate(&stream1) ;
    cudaStreamCreate(&stream2) ;

    cout << "Input File 1: " << argv[1] << endl ; 
    cout << "Input File 2: " << argv[2] << endl ; 
    cout << "Output File:" << argv[3] << endl ; 

    // read both the input matrices
    read_and_send_to_gpu(argv[1], a_dev, stream1) ; 
    read_and_send_to_gpu(argv[2], b_dev, stream2) ;
    cout << "Read Input \n" ;

    // wait for data transfer to complete before launching kernel 
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    auto end_read = std::chrono::high_resolution_clock::now();
    auto duration_read = std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read); // Calculate the duration in milliseconds
    std::cout << "Time taken [READ]: " << duration_read.count() << " milliseconds" << std::endl; // Print the duration in milliseconds

    cout << "Doing Matrix Multiplication \n" ;     
    
    // allocate space for output on the GPU
    // will be initialized in the GPU by zeros
    uint32_t *c_dev ;
    bool* non_zero_output_dev ;
    cudaMalloc(&c_dev, input_n * input_n * sizeof(uint32_t));
    cudaMalloc(&non_zero_output_dev, input_blocks_per_row * input_blocks_per_row * sizeof(bool));
    // cudaMemset((void *)non_zero_output_dev, 0, input_blocks_per_row * input_blocks_per_row * sizeof(bool));
    
    int threads_per_block_x = block_size, threads_per_block_y = block_size ; 
    int blocks_per_grid_x = (input_n + threads_per_block_x - 1 ) / threads_per_block_x ; // if dimension is not a multiple of the number of threads per block than create an additional block
    int blocks_per_grid_y = (input_n + threads_per_block_y - 1) / threads_per_block_y ;

    cout << "GRID DIM " << blocks_per_grid_x << " " << blocks_per_grid_y << "\n" ;
    cout << "BLOCK DIM " << threads_per_block_x << " " << threads_per_block_x << "\n" ;

    dim3 threads_per_block(threads_per_block_x, threads_per_block_y) ;
    dim3 blocks_per_grid(blocks_per_grid_x, blocks_per_grid_y);

    auto start_mult = std::chrono::high_resolution_clock::now();
    
    sparse_matrix_mult_kernel<<<blocks_per_grid, threads_per_block>>>(a_dev, b_dev, c_dev, non_zero_output_dev, input_n, input_m);
    cudaDeviceSynchronize() ;
    
    auto end_mult = std::chrono::high_resolution_clock::now();
    auto duration_mult = std::chrono::duration_cast<std::chrono::milliseconds>(end_mult - start_mult); // Calculate the duration in milliseconds
    std::cout << "Time taken [MULT]: " << duration_mult.count() << " milliseconds" << std::endl; // Print the duration in milliseconds

    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    // // wait for kernel to complete
    cout << "Matrix Multiplication Done \n" ; 


    cout << "Writing Output \n" ;
    // copy output data from the GPU to the CPU
    auto start_copy = std::chrono::high_resolution_clock::now();
    
    uint32_t* c = (uint32_t *)malloc(input_n * input_n * sizeof(uint32_t));
    bool* non_zero_output = (bool *)malloc(input_blocks_per_row * input_blocks_per_row * sizeof(bool));
    cudaMemcpy(c, c_dev, input_n * input_n * sizeof(uint32_t), cudaMemcpyDeviceToHost );
    cudaMemcpy(non_zero_output, non_zero_output_dev, input_blocks_per_row * input_blocks_per_row * sizeof(bool), cudaMemcpyDeviceToHost);
    
    auto end_copy = std::chrono::high_resolution_clock::now();
    auto duration_copy = std::chrono::duration_cast<std::chrono::milliseconds>(end_copy - start_copy); // Calculate the duration in milliseconds
    std::cout << "Time taken [copy]: " << duration_copy.count() << " milliseconds" << std::endl; // Print the duration in milliseconds

    // print_mat_2(c);

    auto start_write = std::chrono::high_resolution_clock::now();
    write_output(argv[3], c, non_zero_output, input_n, input_n);
    cout << "Output Written \n" ;
    auto end_write = std::chrono::high_resolution_clock::now();
    auto duration_write = std::chrono::duration_cast<std::chrono::milliseconds>(end_write - start_write); // Calculate the duration in milliseconds
    std::cout << "Time taken [write]: " << duration_write.count() << " milliseconds" << std::endl; // Print the duration in milliseconds

    
    cudaFree(c_dev); // free unified memory used for output on GPU 
    cudaFree(non_zero_output_dev);
    free(c); // free memory used for output on the CPU

    return 0 ;
}