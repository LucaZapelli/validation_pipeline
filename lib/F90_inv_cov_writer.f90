program MAT_WRITER
        use healpix_types
        implicit none

        integer :: i, j, npixs, ord, pol
        integer :: unit = 10
        character(len=256) :: run_path ='runs/mcmc_IQU-NU_dense/', path
        character(len=256) :: fname
        character(len=256), allocatable, dimension(:) :: filenames
        real(dp), allocatable, dimension(:,:) :: mat        

        allocate(filenames(45))
        filenames(1) = 'Inv_cov_20000000000.0_20000000000.0'
        filenames(2) = 'Inv_cov_20000000000.0_30000000000.0'
        filenames(3) = 'Inv_cov_20000000000.0_40000000000.0'
        filenames(4) = 'Inv_cov_20000000000.0_85000000000.0'
        filenames(5) = 'Inv_cov_20000000000.0_95000000000.0'
        filenames(6) = 'Inv_cov_20000000000.0_145000000000.0'
        filenames(7) = 'Inv_cov_20000000000.0_155000000000.0'
        filenames(8) = 'Inv_cov_20000000000.0_220000000000.0'
        filenames(9) = 'Inv_cov_20000000000.0_270000000000.0'
        filenames(10) = 'Inv_cov_30000000000.0_30000000000.0'
        filenames(11) = 'Inv_cov_30000000000.0_40000000000.0'
        filenames(12) = 'Inv_cov_30000000000.0_85000000000.0'
        filenames(13) = 'Inv_cov_30000000000.0_95000000000.0'
        filenames(14) = 'Inv_cov_30000000000.0_145000000000.0'
        filenames(15) = 'Inv_cov_30000000000.0_155000000000.0'
        filenames(16) = 'Inv_cov_30000000000.0_220000000000.0'
        filenames(17) = 'Inv_cov_30000000000.0_270000000000.0'
        filenames(18) = 'Inv_cov_40000000000.0_40000000000.0'
        filenames(19) = 'Inv_cov_40000000000.0_85000000000.0'
        filenames(20) = 'Inv_cov_40000000000.0_95000000000.0'
        filenames(21) = 'Inv_cov_40000000000.0_145000000000.0'
        filenames(22) = 'Inv_cov_40000000000.0_155000000000.0'
        filenames(23) = 'Inv_cov_40000000000.0_220000000000.0'
        filenames(24) = 'Inv_cov_40000000000.0_270000000000.0'
        filenames(25) = 'Inv_cov_85000000000.0_85000000000.0'
        filenames(26) = 'Inv_cov_85000000000.0_95000000000.0'
        filenames(27) = 'Inv_cov_85000000000.0_145000000000.0'
        filenames(28) = 'Inv_cov_85000000000.0_155000000000.0'
        filenames(29) = 'Inv_cov_85000000000.0_220000000000.0'
        filenames(30) = 'Inv_cov_85000000000.0_270000000000.0'
        filenames(31) = 'Inv_cov_95000000000.0_95000000000.0'
        filenames(32) = 'Inv_cov_95000000000.0_145000000000.0'
        filenames(33) = 'Inv_cov_95000000000.0_155000000000.0'
        filenames(34) = 'Inv_cov_95000000000.0_220000000000.0'
        filenames(35) = 'Inv_cov_95000000000.0_270000000000.0'
        filenames(36) = 'Inv_cov_145000000000.0_145000000000.0'
        filenames(37) = 'Inv_cov_145000000000.0_155000000000.0'
        filenames(38) = 'Inv_cov_145000000000.0_220000000000.0'
        filenames(39) = 'Inv_cov_145000000000.0_270000000000.0'
        filenames(40) = 'Inv_cov_155000000000.0_155000000000.0'
        filenames(41) = 'Inv_cov_155000000000.0_220000000000.0'
        filenames(42) = 'Inv_cov_155000000000.0_270000000000.0'
        filenames(43) = 'Inv_cov_220000000000.0_220000000000.0'
        filenames(44) = 'Inv_cov_220000000000.0_270000000000.0'
        filenames(45) = 'Inv_cov_270000000000.0_270000000000.0'
        
        path = trim(run_path)

        do i = 1, 45
                fname = trim(path) // trim(filenames(i)) // '.txt'
                open(unit, file=trim(fname), status='old')
                read(unit,*) npixs
                read(unit,*) ord
                read(unit,*) pol
                allocate(mat(npixs,npixs))
                read(unit,*) mat
                close(unit)
                unit = unit + 1

                fname = trim(path) // trim(filenames(i)) // '.unf'               
                open(unit, file=trim(fname), form='unformatted')
                write(unit) npixs
                write(unit) ord
                write(unit) pol
                do j=1, npixs
                        write(unit) mat(:,j)
                end do
                close(unit)
                unit = unit + 1

                deallocate(mat)

                fname = trim(path) // 'Sqrt_' // trim(filenames(i)) // '.txt'
                open(unit, file=trim(fname), status='old')
                read(unit,*) npixs
                read(unit,*) ord
                read(unit,*) pol
                allocate(mat(npixs,npixs))
                read(unit,*) mat
                close(unit)
                unit = unit + 1

                fname = trim(path) // 'Sqrt_' // trim(filenames(i)) // '.unf'
                open(unit, file=trim(fname), form='unformatted')
                write(unit) npixs
                write(unit) ord
                write(unit) pol
                do j=1, npixs
                        write(unit) mat(:,j)
                end do
                close(unit)
                unit = unit + 1

                deallocate(mat)

        end do        


end program MAT_WRITER
