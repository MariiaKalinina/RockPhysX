!---------------
	subroutine green_integrand(cc_tens,g, Aijkl)
	USE MSFLIB
	use glbls
	implicit double complex(a-h,o-z)
	real(8):: g(3,3,3,3), n_mn(3,3), cc_tens(3,3,3,3)
	real(8)::  lam(3,3), lam_inv(3,3)
	integer(4) :: ierr, det
	real(8) n1, n2, n3, theta, phi
	real(8) n_m(3), n_n(3)
	   	
    real(8):: Aijkl(3,3,3,3)
	integer(4):: i,j,k,l
	
	call GAUSS(func)
	! gauss-legendre quadrature integrarion 
    	
	! initialize variables	
    n1=0.; n2=0.; n3=0.
        	    	    
    n1= n1 + 1/a1 * sin(theta)*cos(phi)
    n2= n2 + 1/a2 * sin(theta)*sin(phi)
    n3= n3 + 1/a3*cos(phi)

    !print *, n1, n2, n3

    n_mn = 0.
    n_mn = n_mn + matmul(reshape((/n1, n2, n3/), (/3,1/)), reshape((/n1, n2, n3/), (/1,3/)))
    !print *, n_mn
	
    ! problem!!!
    !do i=1,3
        !print '(1x, 10f6.2)', n_mn(i,:)
    !end do
	
    !lam = 0.
    do i=1,3
        do j=1,3
            do k=1,3
                do l=1,3
                    lam(i,k) = lam(i,k) + cc_tens(i,j,k,l)*n_mn(j,l)
                enddo
            enddo
        enddo
    enddo
	  

    call invert3x3(lam, ierr, lam_inv)
    
    
    !Aijkl = 0.
    do i=1,3
        do j=1,3
            do k=1,3
                do l=1,3
                    Aijkl(i,j,k,l) = (-1/(4*pi))*lam_inv(i,k)*n_mn(j,l)*sin(theta)
                enddo
            enddo
        enddo
    enddo
        
        
	return
	end
	
!-------------------------------------------------	