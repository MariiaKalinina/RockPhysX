!---------------
    module globals_perm
real(8) AK(19), AK_init(19),theta(19),fun(19), AK_t(19), weight(19)
!integer N
real(8) x1,x2,ar_grains_elect_data, clay_cond_data
    end module globals_perm
!!!!!
program main
implicit none

integer ind_friab, nn,ind_chaos(3),i,N_fr
parameter(nn = 8)
real(8) x(nn)
real(8) P(3,3),P_matr(3,3),step,friab
real(8) AR_cracks, crack_porosity,AR_grains,AR_pores,pore_porosity

!--------
ind_friab = 12  !GSA
AR_grains = 1
AR_cracks = 0.001
crack_porosity = 0
AR_pores = 0.001
pore_porosity = 15
ind_chaos = 0
ind_chaos(3) = 0    !cracks  are randomly oriented


P_matr = 0.
do i = 1,3
P_matr(i,i) = 3
end do

x(1) = 3	!clay cond. artificial values
x(2) = 0.025	!4e-4	!miliDarcy,	fluid patch perm
x(3) = AR_cracks
x(5) = crack_porosity
x(6) = AR_grains
x(7) = AR_pores
x(8) = pore_porosity

N_fr = 100
step = 1./N_fr
open(1,file = 'LP_GSA_AR0_001.txt')
write(1,'(1x,10(1x,a12))')'friab', 'P(1,1)','P(2,2)', 'P(3,3)'
!------
do i = 1,N_fr+1
friab = (i - 1)*step
x(4) = friab

call forward_ELECT(nn,x,P_matr,AR_cracks, crack_porosity,AR_grains,AR_pores,pore_porosity,ind_chaos,ind_friab,P)
write(1,'(1x,10(1x,f12.6))')friab, P(1,1),P(2,2), P(3,3)
write(*,'(1x,10(1x,f12.6))')friab, P(1,1),P(2,2), P(3,3)
end do

close(1)

stop
end
!______    

subroutine forward_ELECT(nn,x,P_matr,AR_cracks, crack_porosity,AR_grains,AR_pores,pore_porosity,ind_chaos,ind_friab,P)
!subroutine Perm(N_comp,N, lambda, AR_dim, tet_dim, phi_dim, poros_dim, friab, lambda_eff)
! N = 1 - is matrix
! N from 2 to N are inclusions having different shape and orientation
! N_comp is the number of components having different permeability, for one fluid N = 2
	use globals_perm

		implicit real*8(a-h,o-z)
		parameter(N_comp = 2,N=3)
dimension x(nn), xguess(nn), xlb(nn), xub(nn)
	dimension xscale(nn),iparam(7),rparam(7)

real(8) lambda(3,3,N), AR_dim(N),tet_dim(N), phi_dim(N),poros_dim(N),friab,rho_fl(20)
real(8) lambda_eff(3,3), P(3,3),P_matr(3,3)	!,eta_fluid(N_comp)

!--------------------------------------
real(8) F(N), sumc(3,3), sumz(3,3),sumz_inv(3,3),lambda0(3,3), bbeta(3,3),sumc_i_rot(3,3),sumz_i_rot(3,3),lambda_i(3,3)
real(8) sumc_i(3,3),sumz_i(3,3),lambda_V(3,3), lambda_R(3,3), PP(3,3)
integer ind_chaos(N),ind_friab

!write(*,*)'IN FCN'
AR_grains = 1.	!like for elasticity

!x(1) = 1e-4	!clay cond.
!x(2) = 5.	!4e-4	!miliDarcy,	fluid patch perm
!x(3) = AR_cracks
!x(4) = friab
!x(5) = crack_porosity
!x(6) = AR_grains
!x(7) = AR_pores
!x(8) = pore_porosity

poros_dim(1) = 100.-(x(5)+x(8))	!matrix
poros_dim(2) = x(8)	!pore porosity
poros_dim(3) = x(5)	!crack porosity


!------end new-----------------

lambda_eff = 0.
tet_dim = 0.
phi_dim = 0.

unity = 1.
pi = 4.*datan(unity)
lambda = 0.

do i = 1,2
if(i==1)then
lambda(1,1,i) = P_matr(1,1)
lambda(2,2,i) = P_matr(2,2)
lambda(3,3,i) = P_matr(3,3)
end if

if(i==2)then
lambda(1,1,i) = x(2)
lambda(2,2,i) = x(2)
lambda(3,3,i) = x(2)
end if

end do

do i = 1,3
lambda(i,i,3) = x(2)
end do

friab = X(4)
!amu = x(9)

!!pause


!---------- end reading------------------
vcc = poros_dim(2)+poros_dim(3)

	if(N_comp==2)then	! Calculation of comparison body
do i = 1,3
do j = 1,3
!lambda0(i,j) = lambda(i,j,1)*(1.-friab)+lambda(i,j,2)*friab
!goto 88

lambda_R(i,j) = 1./(vcc/100./lambda(i,j,2)+(1.-vcc/100.)/lambda(i,j,1))
lambda_V(i,j) = vcc/100.*lambda(i,j,2)+(1.-vcc/100.)*lambda(i,j,1)

if(ind_friab==17)lambda0(i,j) = lambda_r(i,j)*(1.-friab*vcc/100.)+lambda_v(i,j)*friab*vcc/100.
!if(ind_friab==17)lambda0(i,j) = lambda(i,j,1)*(1.-friab)+lambda(i,j,2)*friab

!cc(i9,j9) = friab*vcc*cc_r(i9,j9)+(1-friab*vcc)*cc_v(i9,j9)  
if(ind_friab==12)lambda0(i,j) = lambda(i,j,1)*(1.-friab)+lambda(i,j,2)*friab

88 continue

end do
end do
!write(*,*)akof*lambda0(1,1),akof*lambda0(2,2),akof*lambda0(3,3)
	end if
	
	!write(*,*)'lambda0 '
!write(*,'(1x,3(1x,e12.6))')((lambda0(ii,j), j = 1,3),ii = 1,3)
!pause

	sumc = 0.
sumz = 0.
!-------------------------------------
! Calculation of formation factors

!alpha = AR_dim(1)
alpha = X(6)	!AR grains


	if(alpha.lt.1.)then
	 t1 = alpha**2 
	 t2 = 1/t1
      t4 = dsqrt(t2-1)
      t5 = atan(t4)
      t8 = t4**2
      Form = t2*(t4-t5)/t8/t4
	  end if

	  if(alpha.gt.1.)then
	   t1 = alpha**2 
	   t2 = 1/t1
      t4 = dsqrt(1-t2)
      t6 = dlog(1+t4)
      t9 = dlog(1-t4)
      t13 = t4**2
      Form = t2*(t6/2-t9/2-t4)/t13/t4
	  end if

	  if(alpha.eq.1.)Form=1./3.

!	 write(*,*)' F (depolarization factor) = ',FORM

	
 	F(1)=Form
!---------------------------------------
do i = 2,N
!alpha = AR_dim(i)
if(i==2)alpha = X(7)	!ar_pores
if(i==3)alpha = X(3)	!ar_cracks

	if(alpha.lt.1.)then
	 t1 = alpha**2 
	 t2 = 1/t1
      t4 = dsqrt(t2-1)
      t5 = atan(t4)
      t8 = t4**2
      Form = t2*(t4-t5)/t8/t4
	  end if

	  if(alpha.gt.1.)then
	   t1 = alpha**2 
	   t2 = 1/t1
      t4 = dsqrt(1-t2)
      t6 = dlog(1+t4)
      t9 = dlog(1-t4)
      t13 = t4**2
      Form = t2*(t6/2-t9/2-t4)/t13/t4
	  end if

	  if(alpha.eq.1.)Form=1./3.

!	 write(*,*)' F (depolarization factor) = ',FORM

	
 	F(i)=Form
	end do
!------------------------------------------

!-----------------------------
sumc = 0.
sumz = 0.

sumc_i = 0.
sumz_i = 0.

do i = 1,N
DO I9 = 1,3
DO J9 = 1,3
if(i==1)LAMBDA_I(I9,J9) = LAMBDA(I9,J9,1)
if(i.gt.1)LAMBDA_I(I9,J9) = LAMBDA(I9,J9,2)

END DO
END DO

F_i = F(i)

call parts(lambda0,lambda_i,F_i,sumc_i,sumz_i)
!write(*,*)'i = ',i
!write(*,*)'sumc_i'
!write(*,*)sumc_i

!write(*,*)'sumz_i'
!write(*,*)sumz_i
!!pause

if(ind_chaos(i) == 1)then

do ii = 1,3
sumc_i_rot(ii,ii)= (sumc_i(3,3)+sumc_i(1,1)+sumc_i(2,2))/3.
sumz_i_rot(ii,ii)= (sumz_i(3,3)+sumz_i(1,1)+sumz_i(2,2))/3.


end do

!write(*,*)'sumc_i_rot '
!write(*,'(1x,3(1x,e12.6))')((sumc_i_rot(ii,j), j = 1,3),ii = 1,3)
!pause

goto 1515

end if

tet_rad = tet_dim(i)/180.*pi
phi_rad = phi_dim(i)/180.*pi
psi_rad = 0.


call beta_rot_t(tet_rad, phi_rad, psi_rad, bbeta)

!write(*,*)'beta'
!write(*,*)bbeta

call tens_rot_perm(sumc_i,sumc_i_rot,bbeta)
call tens_rot_perm(sumz_i,sumz_i_rot,bbeta)

!write(*,*)'sumc_i_rot'
!write(*,*)sumc_i_rot

!write(*,*)'sumz_i_rot'
!write(*,*)sumz_i_rot

1515 continue
do i9 = 1,3
do j9 = 1,3
sumc(i9,j9) = sumc(i9,j9)+poros_dim(i)/100.*sumc_i_rot(i9,j9)
sumz(i9,j9) = sumz(i9,j9)+poros_dim(i)/100.*sumz_i_rot(i9,j9)


end do	!j9
end do	!i9
!write(*,*)'sumc '
!write(*,'(1x,3(1x,e12.6))')((sumc(ii,j), j = 1,3),ii = 1,3)
!pause

!write(*,*)'sumz '
!write(*,'(1x,3(1x,e12.6))')((sumz(ii,j), j = 1,3),ii = 1,3)
!pause

end do	!i

!write(*,*)'sumz FOR INV '
!write(*,'(1x,3(1x,e12.6))')((sumz(ii,j), j = 1,3),ii = 1,3)
!pause

call F1111(sumz,sumz_inv,ind_err)

!write(*,*)'sumz_inv '
!write(*,'(1x,3(1x,e12.6))')((sumz_inv(ii,j), j = 1,3),ii = 1,3)
!pause

call matrix_product_3(sumc,sumz_inv,lambda_eff)
P = lambda_eff

!P=P*akof	!*1e-6	!miliDarcy


return
end
!-------------------------------------
subroutine parts(lambda0,lambda_i,F_i,sumc_i,sumz_i)
! This program calculates the parts of nominator and denominator for effective permeability
! for the i-th component
			implicit real*8(a-h,o-z)
real(8), intent(in):: lambda0(3,3),lambda_i(3,3),F_i
real(8), intent(out):: sumc_i(3,3), sumz_i(3,3)

dimension skob(3,3),F_ten(3,3),AI(3,3),S1(3,3),S2(3,3),S3(3,3),S3_inv(3,3),S4(3,3)

F_ten = 0.
AI = 0.

F_ten(1,1) = (1.-F_i)/2.
F_ten(2,2) = F_ten(1,1)
F_ten(3,3) = F_i

do i = 1,3
AI(i,i) = 1.
end do

do i9 = 1,3
do j9 = 1,3
skob(i9,j9) = AI(i9,j9)-F_ten(i9,j9)
end do
end do

!write(*,*)'skob'
!write(*,*)skob


call matrix_product_3(lambda0,skob,S1)
call matrix_product_3(lambda_i,F_ten,S2)

S3 = S1+S2

call F1111(S3, S3_inv, ind_err)

sumz_i = S3_inv

call matrix_product_3(lambda_i,S3_inv,sumc_i)

return
end
!********************************

!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*********************************************	
          	subroutine beta_rot_t(TET,FI,PSI,BETA)
				 		!USE MSFLIB

!*_________________________________________________________
!*       rotation matrix 3*3
!*_________________________________________________________
	        implicit real*8(a-h,o-z)
                dIMENSION BETA(3,3)

	

      betac1=dcos(tet)
	betas1=dsin(tet)
	betac2=dcos(psi)
	Betas2=dsin(psi)
	betac3=dcos(fi)
	betas3=dsin(fi)

	Beta(1,1)=betac2*betac3-betac1*betas2*betas3
	beta(1,2)=betas2*betac3+betac1*betac2*betas3
	beta(1,3)=betas1*betas3
	
	beta(2,1)=-betac2*betas3-betac1*betas2*betac3
	beta(2,2)=-betas2*betas3+betac1*betac2*betac3
	beta(2,3)=betas1*betac3
	
	beta(3,1)=betas1*betas2
	beta(3,2)=-betas1*betac2
	beta(3,3)=betac1

	 
	return
	END    

!********************************************* 

!*********************************************                      
         SUBROUTINE tens_rot_perm(SS,SSH,beta)
		 	 		USE MSFLIB

!*______________________________________________________
!*        rotation of 2th rank tensor         
!*______________________________________________________
      implicit real*8(a-h,o-z)
      DIMENSION SS(3,3),SSH(3,3),BETA(3,3)
	integer(4) control, newcontrol,status

	
unity = 1.
pi = 4.*datan(unity)

!       call beta_rot(theta,phi,beta)


      DO 1 I=1,3
      DO 1 J=1,3  
      
      SSH(I,J)=0.
      DO 1 K=1,3
      DO 1 L=1,3            


      SSH(I,J)=SSH(I,J)+BETA(I,K)*BETA(J,L)*SS(K,L)

1	CONTINUE

      RETURN
      END                             
!****************************************************
  

!*____________________________________________________        
!******************************
               SUBROUTINE F1111(A,A1,ind_err)
			   	 		
!*____________________________________________________________
!*         inversion of 3*3 matrix
!*____________________________________________________________
                       implicit real*8(a-h,o-z)
!		common/error/ind_err

               DIMENSION A(3,3),A1(3,3)

			   	

               DET=A(1,1)*A(2,2)*A(3,3)
               DET=DET+A(1,3)*A(2,1)*A(3,2)+A(1,2)*A(2,3)*A(3,1)
               DET=DET-A(2,2)*A(1,3)*A(3,1)-A(1,1)*A(3,2)*A(2,3)-A(3,3)*A(2,1)*A(1,2)
               IF (dabs(DET).le.1d-40) THEN
		ind_err=12
		goto 888
			
               ELSE
               GO TO 1
               END IF

1              AM11=A(3,3)*A(2,2)-A(2,3)*A(3,2)
               A1(1,1)=AM11/DET

               AM12=-(A(2,1)*A(3,3)-A(3,1)*A(2,3))
               A1(2,1)=Am12/det

              AM13=A(2,1)*A(3,2)-A(3,1)*A(2,2)
              A1(3,1)=AM13/DET

              AM21=-(A(1,2)*A(3,3)-A(3,2)*A(1,3))
              A1(1,2)=AM21/DET

              AM22=A(1,1)*A(3,3)-A(3,1)*A(1,3)
              A1(2,2)=AM22/DET

              AM23=-(A(1,1)*A(3,2)-A(3,1)*A(1,2))
              A1(3,2)=AM23/DET

              AM31=A(1,2)*A(2,3)-A(2,2)*A(1,3)
              A1(1,3)=AM31/DET

              AM32=-(A(1,1)*A(2,3)-A(2,1)*A(1,3))
              A1(2,3)=AM32/DET

              AM33=A(1,1)*A(2,2)-A(1,2)*A(2,1)
              A1(3,3)=AM33/DET
 888     continue    

 RETURN
              END
!****************************
subroutine matrix_product_3(A,B,C)

 implicit real*8(a-h,o-z)
real(8), intent(in):: A(3,3),B(3,3)
real(8), intent(out)::C(3,3)

do i = 1,3
do j = 1,3
C(i,j) = 0.
do k = 1,3
C(i,j) = C(i,j)+A(i,k)*B(k,j)
end do
end do
end do

return
end