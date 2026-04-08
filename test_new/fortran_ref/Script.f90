module glbls
implicit none
real(8) a1, a2, a3, pi
end module glbls
!---------------------------------
program main
use glbls
implicit none

real(8) cc(6,6),greent(3,3,3,3)

a1 = 100.
a2 = aa1
a3 = 1.

pi=3.14
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

SUBROUTINE GREEN(cc,greent)
use glbls
implicit none

real(8) cc(6,6),g(3,3,3,3),greent(3,3,3,3)
integer i,j,kk,l

!-------
g = 0.
call green_integrand(cc,g)
	greent = g
!--------------

return
end


!---------------
	 subroutine green_integrand(cc,g)
	 !DEC$ ATTRIBUTES DLLEXPORT :: green_anal_11
	 		!USE MSFLIB
	 		use glbls

	!implicit complex(16)(a-h,o-z)
	implicit double complex(a-h,o-z)

	real(8) cc(6,6),c11,c12,c13,c33,c44,c22,c23,c55,c66,eps
	real(8) g(3,3,3,3)
	real(8) n1, n2, n3, theta, phi
	real(8) n_m(1, 3), n_n(3, 1), n_mn(3, 3)
	
	c11=cc(1,1)
	c12=cc(1,2)
	c13=cc(1,3)
	c33=cc(3,3)
	c44=cc(4,4)
	c22=cc(2,2)
	c23=cc(2,3)
	c55=cc(5,5)
	c66=cc(6,6)
	
	theta=0
	phi=1.57
	
	n1=1/a1 * sin(theta)*cos(phi)
	n2=1/a2 * sin(theta)*sin(phi)
	n3=1/a3*cos(phi)
	
	n_m(1,1)=n1
	n_m(1,2)=n2
	n_m(1,3)=n3
	
	n_n(1,1)=n1
	n_n(2,1)=n2
	n_n(3,1)=n3
	
	n_mn=matmul(n_m, n_n)
	print *, n_mn
	

	return
	end
