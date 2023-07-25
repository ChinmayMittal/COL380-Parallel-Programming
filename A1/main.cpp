#include<iostream>
#include<fstream>
#include <assert.h>
#include <vector>
#include<chrono>
#include<omp.h>
#include<algorithm>
#include"library.hh"

int n, m, num_blocks_in_a_row; 

class Block{
    public:
        std::vector<int> arr ; // store the block as a row m*m elements
        int m ; // side of the block

        Block()
        {
            m = 0 ; 
        } 

        void create_block( int m ){
            if( this->m!=0 ){
                // already allocated just clear memory
                std::fill(arr.begin(), arr.end(), 0) ;
                return ;
            }
            // freshly allocate memory
            this->m = m ; 
            this->arr = std::vector<int>(m*m, 0); // block, stored as a row
        }

        void set_value( int i, int j, int value ){
            arr[i*m+j] = value ;
        }

        int get_value( int i, int j ){
            if( m == 0 ){
                // zero block not stored
                return 0 ; 
            }else{
                return arr[i*m+j] ;
            }
        }

        void add_block( Block &b ){
            if(b.m==0){
                return ;
            }
            int total = m*m ; 
            for( int i = 0 ; i < total ; i ++ )
            {
                arr[i] = Outer(arr[i], b.arr[i]) ;
            }
        }

        bool check_zero(){
            int total = m*m ;
            for( int i = 0 ; i < total ; i ++ )
            {
                if(arr[i] != 0 ){
                   return false ;
                }
            }
            return true  ; 
        }

} ; 


Block mult_blocks( Block &a, Block &b){
    int size = a.m ; 
    Block ans ; 
    ans.create_block(size) ; 
    for( int i = 0 ; i < size ; i ++ ){
        for( int j = 0 ; j < size ; j ++ ){
            for( int k = 0 ; k < size ; k ++ ){
                int value = Outer(Inner(a.get_value(i, k), b.get_value(k, j)), ans.get_value(i, j)) ; 
                ans.set_value(i, j, value) ; 
            }
        }
    }
    return ans ; 
}

void mult_blocks_T_acc( Block &a, Block &b, Block&ans){
    int size = a.m ; 
    if(a.m ==0 or b.m  == 0){
        return ;
    }
    // ans.create_block(size) ; 
    // bool non_zero_found = false ;
    for( int i = 0 ; i < size ; i ++ ){
        for( int j = 0 ; j < size ; j ++ ){
            int itemp = i*m, jtemp = j*m ;
            int value = ans.arr[itemp+j] ;
            for( int k = 0 ; k < size ; k ++ ){
                // value = Outer(Inner(a.get_value(i, k), b.get_value(j, k)), value) ; 
                value = Outer(Inner(a.arr[itemp+k], b.arr[jtemp+k] ), value) ; 
                // ans.set_value(i, j, value) ; 
            }
            // if(value!=0) non_zero_found = true ;
            // non_zero_found = non_zero_found or (value) ; 
            // ans.arr[i][j] = value ;
            ans.arr[itemp+j] = value ;
        }
    }
    // return non_zero_found ;
    return ;
}

void read_input(const char* filename, std::vector<std::vector<Block>>&a){

    int  k, i, j ;
    char temp ; 

    std::ifstream rf(filename, std::ios::in | std::ios::binary); 
  
    rf.read((char *)&n, sizeof(n)) ;
    rf.read((char *)&m, sizeof(m)) ; 
    rf.read((char *)&k, sizeof(k)) ; 

    num_blocks_in_a_row = n/m ;

    std :: cout << "Input: " << n << " " << m << " " << k << std::endl ; 
    int blocks_read = 0 ; 
    a = std::vector<std::vector<Block>>(n/m, std::vector<Block>(n/m)) ; 
    
    while(blocks_read < k){
        rf.read((char *)&i, sizeof(i)) ; 
        rf.read((char *)&j, sizeof(j)) ; 
        // std::cout << i << " " << j << std::endl ; 
        a[i][j].create_block(m) ; 
        a[j][i].create_block(m) ; 
        for(int blockrow = 0 ; blockrow < m ; blockrow++){
            for( int blockcol = 0 ; blockcol < m ; blockcol ++ ){
            rf.read((char *)&temp, sizeof(temp)) ; // matrix values are one byte
            // std::cout << int(temp) << " " ; 
            a[i][j].set_value(blockrow, blockcol, int(temp)) ; 
            a[j][i].set_value(blockcol, blockrow, int(temp)) ; 
            }
        }
        // std :: cout << std :: endl ; 
        blocks_read ++ ; 

    }
    rf.close() ; 
    // std::cout << "End " << blocks_read << "\n" ; 


}

void check_output(const char* filename, std::vector<std::vector<Block>>&out){
    int n_out, m_out, k_out ;
    std::ifstream rf(filename, std::ios::in | std::ios::binary); 
    rf.read((char *)&n_out, sizeof(n_out)) ;
    rf.read((char *)&m_out, sizeof(m_out)) ; 
    rf.read((char *)&k_out, sizeof(k_out)) ; 
    std::cout <<"Output: " <<  n_out << " " << m_out << " " << k_out << std::endl ;    
    int blocks_read = 0 ; 
    int i, j ;
    out = std::vector<std::vector<Block>>(n_out/m_out, std::vector<Block>(n_out/m_out)) ; 
    while(blocks_read < k_out){

        rf.read((char *)&i, sizeof(i)) ; 
        rf.read((char *)&j, sizeof(j)) ; 
        // std::cout << i <<  " " << j << ": " ; 

        out[i][j].create_block(m_out) ; 
        for(int blockrow = 0 ; blockrow < m_out ; blockrow ++ ){
            for( int blockcol = 0 ; blockcol < m_out ; blockcol ++ ){
                uint16_t value ;
                rf.read((char *)&value, sizeof(value)) ;
                out[i][j].set_value(blockrow, blockcol, int(value)) ;
                // std::cout << value << " " ; 
            }
        }
        // std::cout << endl ; 
        blocks_read ++ ; 
    }
    rf.close() ; 
}

void write_output(const char* filename, std::vector<std::vector<Block>>&b){
    
    std::ofstream wf ; 
    wf.open(filename, std::ios::out | std::ios::binary ) ;
    int numRows = b.size(), numCols = b[0].size() ;
    int non_zero_count = 0 ; 
    // std :: cout << numRows << " " << numCols << std::endl ; 
    for( int i = 0 ; i < numRows ; i ++ ){
        for( int j = i ; j < numCols ; j ++ ){
            if( ! b[i][j].check_zero()){
                non_zero_count ++ ; 
            }
        }
    } 
    std::cout << "Non Zero Count: " <<  non_zero_count << std::endl ; 
    // write n, m, k
    wf.write((const char *)&n, sizeof(n)) ;
    wf.write((const char *)&m, sizeof(m)) ;
    wf.write((const char *)&non_zero_count, sizeof(non_zero_count)) ;

    for( int i = 0 ; i < numRows ; i ++ ){
        for( int j = i ; j < numCols ; j ++ ){
            if( ! b[i][j].check_zero()){
                // non-zero matrix
                // std::cout << i << " " << j << ": " ;
                wf.write((const char* )&i, sizeof(i)) ;
                wf.write((const char* )&j, sizeof(j)) ;
                for( int block_row = 0 ; block_row < m ; block_row ++ ){
                    for( int block_col = 0 ; block_col <m ; block_col ++ ){

                        int v = b[i][j].get_value(block_row, block_col);
                        uint16_t value = ( v >= (1<<16) ? (1<<16)-1 : v ) ; 
                        wf.write((const char *)&value, sizeof(value)) ;
                        // std::cout << value << " " ; 
                    }
                }
                // std::cout << endl ; 
            }
        }
    } 

    wf.close() ; 


}

bool check_equal( std::vector<std::vector<Block>>&a,  std::vector<std::vector<Block>>&b){
    int arows = a.size(), acols = a[0].size(), brows = b.size(), bcols = b[0].size() ;
    if( arows!=brows || acols!=bcols){
        return false ;
    }

    bool are_equal = true ;
    for( int i = 0 ; i < brows ; i ++ ){
        for( int j = i ; j < bcols ; j ++ ){
            if(a[i][j].m != b[i][j].m){ are_equal = false ; break ;}
            if(a[i][j].m > 0 ){
                for( int block_row = 0 ; block_row < a[i][j].m ; block_row ++ ){
                    for( int block_col = 0 ; block_col < a[i][j].m ; block_col ++ ){
                        if(! (a[i][j].get_value(block_row, block_col) == b[i][j].get_value(block_row, block_col))){
                            are_equal = false ;
                            break ;
                        }
                    }
                }
            }
        }
    }

    return are_equal ;
}

void naive_mat_mul( std::vector<std::vector<Block>>&a,  std::vector<std::vector<Block>>&b ){
    
    // naive single threaded multiplication
    // only compute upper traingular part
    for( int block_row = 0 ; block_row < n/m ; block_row ++ ){
        for( int block_col = block_row ; block_col < n / m ; block_col ++ ){
            b[block_row][block_col].create_block(m) ; 
            for( int k = 0 ; k < n / m ; k ++ ){
                Block partial_prod = mult_blocks(a[block_row][k], a[k][block_col]) ; 
                b[block_row][block_col].add_block(partial_prod) ;
            }
        }
    }

    return ;
}

void compute_output_block(int i, int j, std::vector<std::vector<Block>>&a,  std::vector<std::vector<Block>>&b)
{
    b[i][j].create_block(m) ;
    // bool created = false;
    // Block a[i][k] multiplier with a[j][k] for all values of k and summer
    for( int k = 0 ; k < num_blocks_in_a_row ; k ++ )
    {
        if(a[i][k].m and a[j][k].m)
        {
            // iterate over output_rows and output_columns
            for( int output_r = 0 ; output_r  < m ; output_r ++ )
            {
                for( int output_c = 0 ; output_c < m ; output_c ++ )
                {
                    int tempr = output_r*m;
                    int tempc = output_c* m ;
                    int value = ((k == 0) ? Inner(a[i][k].arr[tempr], a[j][k].arr[tempc]) : b[i][j].arr[tempr+output_c] ) ; // initialization
                    int inner_k = ((k==0) ? 1 : 0 ) ;
                    for( ; inner_k < m ; inner_k ++ )
                    {
                        // block being multiplied by its transpose
                        // a[i][k] * a[j][k]T i output_r row of first matrix multiplied with output_c of second matrix
                        value = Outer(Inner(a[i][k].arr[tempr+inner_k], a[j][k].arr[tempc+inner_k]), value) ;
                    }
                    b[i][j].arr[tempr+output_c] = value ;
                }
            }
        }
    }
    if(b[i][j].check_zero()) {
        b[i][j].m = 0 ;
        b[i][j].arr.clear() ;
        b[i][j].arr.shrink_to_fit() ;
    }
    return ;
}



void implementation_1( std::vector<std::vector<Block>>&a,  std::vector<std::vector<Block>>&b){

    // int number_of_block_rows = a.size(), number_of_block_cols = a[0].size() ; 
    // int number_of_threads= (number_of_block_rows)*(number_of_block_rows+1)/2 ; 
    // split output_block amongst threads
    // const auto processor_count = std::thread::hardware_concurrency() ;
    int processor_count = omp_get_num_procs(), number_of_threads ;
    if( processor_count >= 48 or processor_count <= 0 ) number_of_threads = 8 ;
    else number_of_threads = processor_count ;
    // std::cout << "Number of Processors: " << processor_count << " " << "Threads: " << processor_count << std::endl ;

    #pragma omp parallel num_threads(number_of_threads)
    {
            #pragma omp for
            for( int i = 0 ; i < num_blocks_in_a_row ; i ++ ){
                for( int j = i ; j < num_blocks_in_a_row ; j ++ ){
                    #pragma omp task
                    {
                        compute_output_block(i, j, a, b) ;
                    }
                }
            }
    }

}



void implementation_2( std::vector<std::vector<Block>>&a,  std::vector<std::vector<Block>>&b){
    
    // split output row amongst threads
    int number_of_threads = 8; 
    #pragma omp parallel num_threads(number_of_threads)
    {
            #pragma omp for
            for( int i = 0 ; i < num_blocks_in_a_row ; i ++ ){
                #pragma omp task
                {
                    for( int j = i ; j < num_blocks_in_a_row ; j ++ ){
                        compute_output_block(i, j, a, b) ;
                    
                    }
                }
            }
    }

}

void print_matrix( std::vector<std::vector<Block>>&b)
{
    // writing output to terminal
    for(int i = 0 ; i < n ; i ++ ){
        std :: cout << "Row " << i << " " ; 
        for( int j = 0 ; j < n ; j ++ ){
            int value = b[i/m][j/m].get_value(i%m, j%m) ; 
            if(value)
                std::cout << value << " " ; 
        }
        std::cout << std::endl ; 
    }
}
int main(int argc, char *argv[]){

    assert(argc > 1) ; 
    // input matrix
    std::vector<std::vector<Block>> a ; 
    read_input(argv[1], a) ;
    std::cout << "Read Input\n" ;
    //output matrix
    std::vector<std::vector<Block>> b(n/m, std::vector<Block>(n/m)) ; 

    auto begin = std::chrono::high_resolution_clock::now();
    // naive_mat_mul(a, b) ;// works
    implementation_1(a, b) ; 
    // implementation_2(a, b) ;
    auto end = std::chrono::high_resolution_clock::now();
    float ms = (1e-6 * (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)).count()) ;
    std::cout << "Time Taken: " << ms << " ms\n" ;


    std::vector<std::vector<Block>> output_mat, output_mat2 ; 
    // check_output("./output1", output_mat) ; // read correct answer
    // std::cout << "Read Answer\n" ;
    // write output to binary file
    write_output(argv[2], b) ;
    std::cout << "Write Complete\n" ; 
    // print_matrix(b) ;
    // check_output(argv[2], output_mat2) ; 
    
    // std::cout << "Equal: " << check_equal(output_mat, output_mat2) << std::endl ; 
    return 0 ; 
} 