	
module glbls
    implicit none
    real(8), parameter:: pi=3.14159265
end module glbls

! integrand function f(x,y)
function func(x,y) result(res)
    implicit none
    integer, parameter:: p=16 ! quadruple precisi
    real(kind=p), intent(in):: x, y
    real(kind=p):: res
    res = x**2+y
end function func

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

! Roots of the polynomial obtained by
! Newtow-Raphson method and weights
function gaussquad(n) result(r)
    !external::Legendre, DLegendre
    use glbls 
    implicit none
    integer, parameter:: p=16 ! quadruple precision
    
    real(kind=p), external:: Legendre, DLegendre
    integer :: n
    real(kind=p):: r(2, n),xi,dx,error
    integer :: i, iters
    double precision, parameter:: tolerance=1.D-20
    
    error=0.0
    
    if (n.gt.2) then
        do i = 1, n
            xi = cos(pi*(i-0.25_p)/(n+0.5_p))
            error=10*tolerance
            iters=0
            
            dx=-Legendre(n, xi)/DLegendre(n, xi)
            xi=xi+dx
            iters=iters+1
            !error=abs(dx)
            
            r(1,i) = xi
            r(2,i) = 2/((1-xi**2)*DLegendre(n, xi)**2)
        enddo
    else 
        error=1.0
    end if 
    
    print *, r
end function gaussquad

program GaussLegendreQuadrature
    use glbls 
	implicit none
	integer, parameter:: p=16 ! quadruple precision
	
	real(kind=p):: x, y
	integer::n ! n=polynomial order
    external Legendre, DLegendre
    real(kind=p)::func, gaussquad
    
    real(kind=p):: xi,dx,error
    integer :: iter,iters
    double precision, parameter:: tolerance=1.D-20
    
	real(kind=p):: x1, x2, y1, y2
	real(kind=p), allocatable :: r(:,:) !r=roots of the polynomial
	real(kind=p):: z
	integer:: i, j, k
	
	x=2.0; y=3.40
	n=3
	x1=0; x2=2*pi; y1=0; y2=pi
	
	do i=1,n
		r = gaussquad(i)
		print*, r
	enddo
	
	!z=0.
	do j=1,n
		do k=1,n
			! r(2,:)=вес; r(1,:)=корень полинома
			!z = dot_product(r(2,:),exp((a+b)/2+r(1,:)*(b-a)/2))
			!z = z+((x2-x1)/2*(y2-y1)/2)*(r(2,j)*r(2,k))*func((x2-x1)/2*r(1,j)+(x2+x1)/2, (y2-y1)/2*r(1,k)+(y2+y1)/2)	
			z = z+((x2-x1)/2*(y2-y1)/2)*(r(2,j)*r(2,k))	
		enddo
	enddo

    print*, z
stop
end program GaussLegendreQuadrature


!recursive function factorial(n) result(resf)
    !integer::n
    !integer(kind=8)::fres
    !if (n.eq.0) then
        !fres=1
   ! else
        !fres=n*factorial(n-1)
    !endif
!end function factorial 