module glbls

    implicit none
    real(8) a1, a2, a3, pi
    parameter (pi=3.14159265)
    integer::p
    parameter(p=16)
    
end module glbls
!---------------------------------
program main

    use glbls
    implicit none
    real(p):: cc(6,6),greent(3,3,3,3), tens(3,3,3,3), cc_tens(3,3,3,3)
    integer(4) m2t(3,3)/1,6,5,6,2,4,5,4,6/
    integer(4):: i,j,k,l
    
    a1 = 100.
    a2 = a1
    a3 = 1.
   
    cc=0.
    cc(1,1)=136.
    cc(2,2)= cc(1,1)
    cc(3,3)= cc(1,1)

    cc(1,2)=33.
    cc(1,3)=cc(1,2)
    cc(2,1)=cc(1,2)
    cc(2,3)=cc(1,2)
    cc(3,1)=cc(1,2)
    cc(3,2)=cc(1,2)

    cc(4,4)= 12.
    cc(5,5)=cc(4,4)
    cc(6,6)=cc(4,4)
    
    ! Voight notation: matrix 6x6 to tensor(3,3,3,3)
    do i = 1,3
        do j = 1,3
            do k = 1,3
                do l = 1,3
                    tens(i,j,k,l) = cc(m2t(i,j), m2t(k,l))       
                 enddo
             enddo
         enddo
    enddo

    !print *, tens
    cc_tens = tens
    call GREEN(cc_tens,greent)
    
    !write(1,'(1x,10(1x,f12.6))')cc
    !write(*,'(1x,10(1x,f12.6))')cc

    stop
    end
    !---------------------------------

    subroutine GREEN(cc_tens,g_tensor)
        use glbls
        implicit none
        real(p):: cc(6,6),g(3,3,3,3),greent(3,3,3,3), cc_tens(3,3,3,3) 
        real(p):: g_tensor(3,3,3,3), Akmln(3,3,3,3)
        
        !call identity_4(EYE4)
       
        
        !g=0.    
        !call green_integrand(cc_tens,g)
        !greent = g
        
            call GaussLegendreQuadrature(g_tensor)
            Akmln = g_tensor
            
    return
    end subroutine GREEN

!---------------
! transform matrix(6,6) to tensor(3,3,3,3)
!subroutine mat2tens(cc)
    !use glbls
    !implicit none
    !real(8) cc(6,6), tens(3,3,3,3)
    !integer(4) m2t(3,3)/1,6,5,6,2,4,5,4,6/
    !integer(4):: i,j,k,l

    !print *, cc

    !do i = 1,3
        !do j = 1,3
            !do k = 1,3
                !do l = 1,3
                    !tens(i,j,k,l) = cc(m2t(i,j), m2t(k,l))       
                 !enddo
             !enddo
         !enddo
    !enddo

    !return
    !end subroutine mat2tens
  
!---------------
!inverse matrix(3,3)
subroutine invert3x3(lam, lam_inv)

    use glbls
    implicit none
    
    real(p):: lam(3,3), lamT(3,3), lam_inv(3,3)
    integer(4):: det
    integer(4):: i,j
        
    ! определение миноров матрицы; lamT- транспонированная матрица (3,3)      
    lamT(1,1) = lam(2,2)*lam(3,3) - lam(3,2)*lam(2,3)
    lamT(1,2) = lam(3,2)*lam(1,3) - lam(1,2)*lam(3,3)
    lamT(1,3) = lam(1,2)*lam(2,3) - lam(1,3)*lam(2,2)
    
    lamT(2,1) = lam(2,3)*lam(3,1) - lam(2,1)*lam(3,3)
    lamT(2,2) = lam(1,1)*lam(3,3) - lam(3,1)*lam(1,3)
    lamT(2,3) = lam(2,1)*lam(1,3) - lam(1,1)*lam(2,3)
    
    lamT(3,1) = lam(2,1)*lam(3,2) - lam(2,2)*lam(3,1)
    lamT(3,2) = lam(3,1)*lam(1,2) - lam(1,1)*lam(3,2)
    lamT(3,3) = lam(1,1)*lam(2,2) - lam(1,2)*lam(2,1)
    
    ! определитель матрицы
    det = lamT(1,1)*lam(1,1) + lamT(1,2)*lam(2,1) +  lamT(1,3)*lam(3,1)

    if (det.eq.0) then
        !write (ERROR_UNIT,*) 'Invert failed'
        print*, 'Invert failed. DET=0'
        !return
    endif
    
    !det = 1._dp/det 
    do i=1,3
        do j=1,3
            lam_inv(i,j) = lamT(i,j)*1./det
        enddo
    enddo
    !print *, lam_inv
    
    return
    end subroutine invert3x3              

!---------------
! fourth order identity tensor
subroutine identity_4(EYE4)
use glbls
implicit none

real(8):: EYE2(3,3), EYE4(3,3,3,3)
integer(4):: nDim
integer(4):: i,j,k,l

nDim=3
EYE2=0.d0
EYE2(1,1) = 1.d0
EYE2(2,2) = EYE2(1,1)
EYE2(3,3) = EYE2(1,1)

do i = 1, nDim
    do j = 1, nDim
        do k = 1, nDim
            do l = 1, nDim
                EYE4(i,j,k,l) = (EYE2(i,l) * EYE2(j,k) + EYE2(i,k)*EYE2(j,l))/2.d0
            enddo
        enddo
    enddo
enddo

print *, EYE4

return
end	
!-------------------------------------------------	
!! integrand function f(x,y)
!function func(x,y) result(res)
    !implicit none
    !real(p), intent(in):: x, y
    !real(p):: res
    !res = x**2+y
!end function func

!subroutine green_integrand(cc_tens,g, Aijkl)
function func(phi, theta, cc_comp) result(Aijkl)

    USE MSFLIB
    use glbls
    implicit double complex(a-h,o-z)
    
    real(p), intent(in):: phi, theta
    real(p), intent(in):: cc_comp(3,3,3,3)
   
    real(p):: g(3,3,3,3), n_mn(3,3), cc_tens(3,3,3,3)
    real(p)::  lam(3,3), lam_inv(3,3)
    real(p)::  n1, n2, n3, n_m(3), n_n(3)
    
    external invert3x3
    integer(4) :: det
       	
    real(p):: Aijkl(3,3,3,3)
    integer(4):: i,j,k,l
            	    	    
    n1= 1/a1 * sin(theta)*cos(phi)
    n2= 1/a2 * sin(theta)*sin(phi)
    n3= 1/a3*cos(phi)

    n_mn = 0.
    n_mn = n_mn + matmul(reshape((/n1, n2, n3/), (/3,1/)), reshape((/n1, n2, n3/), (/1,3/)))
    
    lam=0.
    
    do i=1,3
        do j=1,3
            do k=1,3
                do l=1,3
                    lam(i,k) = lam(i,k) + cc_comp(i,j,k,l)*n_mn(j,l)
                enddo
            enddo
        enddo
    enddo
      
    call invert3x3(lam, lam_inv)
    
    Aijkl=0.

    do i=1,3
        do j=1,3
            do k=1,3
                do l=1,3
                    Aijkl(i,j,k,l) = (-1/(4*pi))*lam_inv(i,k)*n_mn(j,l)*sin(theta)
                enddo
            enddo
        enddo
    enddo

end function func

!-------------------------------------------------	

!Recursive generation of the Legendre polynomial of order n/roots	
recursive function Legendre(n, xi) result(Pn)
    implicit none
    integer, parameter:: p=16 ! quadruple precision
    integer::n
    real(kind=p)::xi, Pn

    if (n.eq.0.)then
	    Pn=1.0
    else if (n.eq.1.) then
	    Pn=xi
    else 
        Pn=((2.0*n-1.0)*xi*Legendre(n-1,xi)-(n-1)*Legendre(n-2,xi))/n
	    !Pn=(-(n-1)*xi*Legendre(n-2,xi)+(2*n-1)*xi*Legendre(n-1,xi))/n
    end if
end function Legendre

!Derivative of the Legendre polynomials (for weights calculation)
recursive function DLegendre(n, xi) result(DPn)
    implicit none 
    integer, parameter:: p=16 ! quadruple precision
    integer::n
    real(kind=p)::xi, DPn

    if (n.eq.0.)then
	    DPn=xi*0
    else if (n.eq.1.)then
	    DPn=xi+1
    else 
	    DPn=(n/(xi**2-1.0))*(xi*DLegendre(n,xi)-DLegendre(n-1,xi))
    end if
end function DLegendre

! Roots of the polynomial obtained by Newtow-Raphson method and weights
function gaussquad(n) result(r)
    !external::Legendre, DLegendre
    use glbls 
    implicit none
 
    real(kind=p), external:: Legendre, DLegendre
    integer :: n
    real(kind=p):: r(2, n),xi,dx,error
    integer :: i, iter
    double precision, parameter:: tolerance=1.D-20
    
    error=0.0
    
    if (n.gt.2) then
        do i = 1, n
            xi = cos(pi*(i-0.25_p)/(n+0.5_p))
            error=10*tolerance
            iter=0
            
            dx=-Legendre(n, xi)/DLegendre(n, xi)
            xi=xi+dx
            iter=iter+1
            !error=abs(dx)
            
            r(1,i) = xi
            r(2,i) = 2/((1-xi**2)*DLegendre(n, xi)**2)
        enddo
    else 
        error=1.0
    end if 
    
    print *, r
    
end function gaussquad

subroutine GaussLegendreQuadrature(g_tensor)
    USE MSFLIB
    use glbls 
	implicit none

	real(p):: phi, theta
	real(p):: Aijkl(3,3,3,3), g_tensor(3,3,3,3), func_res(3,3,3,3)
	real(p):: cc_tens(3,3,3,3), cc_comp(3,3,3,3)
	integer:: ii, jj, kk, ll
	
	integer::n ! n=polynomial order
    external Legendre, DLegendre
    real(kind=p)::func, gaussquad
    
    real(kind=p):: xi,dx,error
    integer :: iter
    double precision, parameter:: tolerance=1.D-20
    
	real(kind=p):: x1, x2, y1, y2
	real(kind=p), allocatable:: r(:,:) !r=roots of the polynomial
	integer:: i, j, k
	
	n=3
	x1=0.0; x2=2*pi; y1=0.0; y2=pi
	cc_comp=cc_tens
	
	do i=1,n
		r = gaussquad(i) ! r(1,:) - roots, r(2,:) - weights
	enddo
	
	g_tensor=0.
	
     do ii=1,3
        do jj=1,3
            do kk=1,3
                do ll=1,3
                
                    do j=1,n
		                do k=1,n
		                        func_res = func((x2-x1)/2*r(1,j)+(x2+x1)/2, (y2-y1)/2*r(1,k)+(y2+y1)/2, cc_comp)
		                        
                                g_tensor(ii,jj,kk,ll) = g_tensor(ii,jj,kk,ll)+&
                                ((x2-x1)/2*(y2-y1)/2)*(r(2,j)*r(2,k))*&
                                func_res(ii, jj, kk, ll)
                        enddo
                    enddo
                
                enddo
            enddo
        enddo
    enddo
    	
	!z=0.0
	!do j=1,n
		!do k=1,n
			! r(2,:)- вес; r(1,:)- корень полинома
			!z = dot_product(r(2,:),exp((a+b)/2+r(1,:)*(b-a)/2))			
			!z = z+((x2-x1)/2*(y2-y1)/2)*(r(2,j)*r(2,k))*func((x2-x1)/2*r(1,j)+(x2+x1)/2, (y2-y1)/2*r(1,k)+(y2+y1)/2)	
		!enddo
		
	!enddo
    print*, g_tensor
end subroutine GaussLegendreQuadrature


! tensor Gklmn symmetrised
subroutine symmetrization(g_kmln)

    USE MSFLIB
    use glbls 
	implicit none
	
	real(p):: Akmln(3,3,3,3), g_kmln(3,3,3,3)
	integer(4):: k,l,n,m
	
	g_kmln=0.
	
	do k=1,3
        do l=1,3
            do n=1,3
                do m=1,3
                    g_kmln(k,l,n,m) = 0.25_p*(Akmln(k,l,m,n)+Akmln(m,l,n,k)+&
                    Akmln(k,n,l,m)+Akmln(m,n,l,k))
                enddo
            enddo
        enddo
     enddo

end subroutine symmetrization















