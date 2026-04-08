!---------------
    module globals_perm
real(8) AK(19), AK_init(19),theta(19),fun(19), AK_t(19), weight(19)
!integer N
real(8) x1,x2,ar_grains_elect_data, clay_cond_data
    end module globals_perm
    
!----
    module global_G
    real(8) i5,i6,i25,i26,k81,k82,k822
	real(8) aa1,aa2,aa3  
    
    end module global_G
!!!!!
program main
use global_G
implicit none

integer ind_friab, nn,ind_chaos(3),i,N_fr
parameter(nn = 8)
real(8) x(nn)
real(8) P(3,3),P_matr(3,3),step,friab
real(8) AR_cracks, crack_porosity,AR_grains,AR_pores,pore_porosity

!--------
ind_friab = 12  !GSA
AR_grains = 1
AR_cracks = 0.001    !0.001
crack_porosity = 5
AR_pores = 1.
pore_porosity = 0.
ind_chaos = 0
ind_chaos(3) = 0    !cracks  are aligned


P_matr = 0.
do i = 1,3
P_matr(i,i) = 7.6
end do

x(1) = 7.6	!clay cond. artificial values
x(2) = 0.12	!4e-4	!miliDarcy,	fluid patch perm
x(3) = AR_cracks
x(5) = crack_porosity
x(6) = AR_grains
x(7) = AR_pores
x(8) = pore_porosity

N_fr = 100
step = 1./N_fr
open(1,file = 'LP_GSA_CONC_AR0_001_G_MAPLE.txt')
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
use global_G

		implicit real*8(a-h,o-z)
		parameter(N_comp = 2,N=3)
dimension x(nn), xguess(nn), xlb(nn), xub(nn)
	dimension xscale(nn),iparam(7),rparam(7)

real(8) lambda(3,3,N), AR_dim(N),tet_dim(N), phi_dim(N),poros_dim(N),friab,rho_fl(20)
real(8) lambda_eff(3,3), P(3,3),P_matr(3,3)	!,eta_fluid(N_comp)

!--------------------------------------
real(8) F(N), sumc(3,3), sumz(3,3),sumz_inv(3,3),lambda0(3,3), bbeta(3,3),sumc_i_rot(3,3),sumz_i_rot(3,3),lambda_i(3,3)
real(8) sumc_i(3,3),sumz_i(3,3),lambda_V(3,3), lambda_R(3,3), PP(3,3),G_i(3,3),G(3,3)
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
! i = 1 matrix
aa1 = 1./alpha
aa2 = aa1
aa3 = 1.

call f27000
call green(lambda0,G)	

	
 	G_i = G
!---------------------------------------
do i = 2,N
!alpha = AR_dim(i)
if(i==2)alpha = X(7)	!ar_pores
if(i==3)alpha = X(3)	!ar_cracks

	aa1 = 1./alpha
    aa2 = aa1
    aa3 = 1.
call f27000
!	 write(*,*)' F (depolarization factor) = ',FORM

call green(lambda0,G)	
 	G_i = G
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



call parts_CONC_G(lambda0,lambda_i,G_i,sumc_i,sumz_i)
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
subroutine parts_CONC_G(lambda0,lambda_i,G_i,sumc_i,sumz_i)
use global_G
! This program calculates the parts of nominator and denominator for effective permeability
! for the i-th component
			implicit real*8(a-h,o-z)
real(8), intent(in):: lambda0(3,3),lambda_i(3,3),G_i(3,3)
real(8), intent(out):: sumc_i(3,3), sumz_i(3,3)

real(8) AI(3,3),S1(3,3),S2(3,3),S3(3,3),lambda0_inv(3,3),CONC(3,3),CONC_inv(3,3)
!---------------------
!write(*,*)'lambda0 = '
!write(*,*)lambda0(1,1),lambda0(2,2),lambda0(3,3)
!write(*,*)'lambda_i = '
!write(*,*)lambda_i(1,1),lambda_i(2,2),lambda_i(3,3)
!write(*,*)'F_i = ', F_i
!pause



AI = 0.

do i = 1,3
AI(i,i) = 1.
end do

S1 = lambda_i - lambda0
call matrix_product_3(G_i,S1,S2)
CONC_inv = AI-S2
call F1111(Conc_inv, CONC, ind_err)

sumz_i = CONC
call matrix_product_3(lambda_i,CONC,sumc_i)


return
    end
!********************************
    subroutine green(cc,gg33)
!___________________________________________________________
!        The 2th derivative of the Green tensor (nonsymmetric)
!__________________________________________________________
use global_G
implicit none

real*8 gg33(3,3),cc(3,3)
real*8 t1,t3,t4,t5,t6,t11,t10,t8

gg33 = 0.
t1 = aa1**2
      t3 = dsqrt(t1-1)
      t4 = datan(t3)
      t5 = t1**2
      t8 = t3**2
      t11 = t8**2
      gg33(1,1) = -1/cc(1,1)/t11/t3*(-t1*t4-t8*t3+t5*t4)/2
      
      gg33(2,2) = gg33(1,1)
      
  t1 = aa1**2
      t3 = dsqrt(t1-1)
      t4 = datan(t3)
      t6 = t3**2
      t10 = t6**2
      gg33(3,3) = 1/cc(1,1)/t10/t3*(t1*t4-t6*t3-t4)*t1
    
return
    end
    
!***************************
subroutine green_old(cc,gg33)
!___________________________________________________________
!        The 2th derivative of the Green tensor (nonsymmetric)
!__________________________________________________________
use global_G

implicit real*8(a-h,o-z),integer*2(i-n)

      dimension S3(6,6),xt$(0:1000),xf$(0:1000),cc(3,3)
      dimension cct(3,3),aat$(3,3),gg33(3,3)
      dimension ann$(3,3)
      real*8 mno,mno1$
      !i5,i6,i25,i26
	
	!common/int/i5,i6,i25,i26,k81,k82,k822
	!common/a/aa1,aa2,aa3  
	PI=3.1416
      	
!------------------------------      
      do i=1,3
      do m=1,3
   
      cct(i,m)=cc(i,m)
      gg33(i,m)=0.      
      enddo
      enddo
!--------------------------------      
      
      
        UGU4=1.
!-----------the value of UGU4 shows the number of intervals, into which
!----------the Fi definition domain is divided, according to the medium 
!----------------symmetry (4 for orthotropic and 1 for HIGHER symmetries        
      KUF = 1
      do i = 1,3
      do j = 1,3
      S3(i, j) = 0.
      end do
      end do
      i7=0

3344       do i = 0,k81
       XT$(i) = -1. + 2. / k81 * i
           end do
           
       ht = 2. / k81
             
       DO N1=0,K81
       KUT = 1

3345   do i = 0,K82
       XF$(i) = -1. + 2. / K82 * i
       end do
       
       hf = 2. / K82
       
       DO N2=0,K82
           kr8 = k81
       call f13000(N1,kof,Kr8)
       aks=kof
       kr8 = k82
       call f13000(N2,kof,Kr8)
       aks=aks*kof

                       
       
       IF (KUF.eq.1) THEN
       t$ = I25 * (1. + XT$(N1))
       ELSE

        if(kuf.eq.2) then
        t$=pi/2.+(pi-2.*i5)/2.*xt$(n1)
        else    
        
       t$ = (2.*pi-i5)/2.+i5/2.*xt$(n1)
       endif
       endif
             
       f$ = I26 * (1. + XF$(N2))
      
       snint$ = SIN(t$)
       I7 = I7 + 1
199    format('+',i7,tl22)
             
       an1$ = snint$ * COS(f$) / aa1
       an2$ = snint$ * SIN(f$) / aa2
       an3$ = COS(t$) / aa3
       an12$ = an1$ ** 2
        an22$ = an2$ ** 2
         an32$ = an3$ ** 2       
!--------------------------------------------
      ann$(1,1)=an12$
      ann$(1,2)=an1$*an2$
      ann$(1,3)=an1$*an3$
      ann$(2,2)=an22$
      ann$(2,3)=an2$*an3$
      ann$(3,3)=an32$
      
      do i=1,3
      do j=i,3
      ann$(j,i)=ann$(i,j)
      enddo
      enddo
      
!-------------------------------------------      
      
     
      g$=0.
      
      do m=1,3
      do n=1,3
      
      g$=g$+cct(m,n)*ann$(m,n)
      enddo
      enddo
!----------------------------------------------            
       a1=1./g$
!_________
      
         g$=a1
!--------------------------------
      do 1133 i=1,3
      do 1133 j=1,3

      aat$(i,j)=ann$(i,j)*g$
1133  continue
!---------------------------------------             

         IF (KUF.eq.1) THEN
         piK = I25
         ELSE

      if(kuf.eq.2)then
      pik=(pi-2.*i5)/2.
      
      else
      pik=i25
      endif
      
         END IF
                          
         
         PIK1 = I26
         

         mno = aks * piK * PIK1 * ht * hf / 9. / 4. / PI
         mno1$ = mno * snint$ 
!-------------------------------------------------------
      do 1144 i=1,3
      do 1144 j=1,3
   
      
   
     
      gg33(i,j)=gg33(i,j)-aat$(i,j)*mno1$*UGU4  
      
1144  continue     
                                
                          

      end do

      
15533   XXX=1.
        end do
          
         
          
        IF (KUF.eq.1) THEN
        KUF = 2
        GOTO 3344  
         
         else
         
         if(kuf.eq.2)then
         kuf=3
         goto 3344
         end if
         
         end if
!------------------------end of integration-------
     

55        format(1x,6f9.6)

        ity$ = 0
        KUF=1
!--------------------------------------------------       
      write(*,*)'gg '
	write(*,'(1x,3f9.6)')((gg33(i,j),j=1,3),i=1,3) 
	 
!_____________________
        
        RETURN
    end
!**************************************************
    !*************************************8         
              SUBROUTINE F13000 (KIFA,KOF,KR8)
!*______________________________________________________
!*        Service sub-program for integration
!*_____________________________________________________
                 implicit real*8(a-h,o-z),integer*2(i-n)

         IF ((kifa.EQ.0).OR.(kifa.EQ.kr8)) THEN
         kof = 1
         RETURN
         ELSE
	 skifa=kifa
	 AM=sKIFA/2.
	 aI=aint(am)
         IF (AM.NE.AI) THEN
         kof = 4
         ELSE
         kof = 2
         END IF
         END IF

         RETURN
         END
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
    !----
    !*************************************************
                 SUBROUTINE F27000
!*____________________________________________
!*          the choise of integration net
!*_____________________________________________
use global_G

implicit real*8(a-h,o-z),integer*2(i-n)

          
	!common /a1/ pi
           !COMMON /A/ AA1,AA2,AA3
           
           !integer*2 priz$(10)

           !common /pr/ priz$

           !COMMON /INT/ I5,I6,I25,I26,K81,K82,K822
          
           !REAL*8 I5,I6,I25,I26
           
          PI=3.1416
          !write(*,*)'aa1 = ',aa1
		          !write(*,*)'aa2 = ',aa2
	          !write(*,*)'aa3 = ',aa3

! 		pause
         if (AA3.GT.1) then
           GO TO 103
           ELSE

           if (aa1.LE.10) then
           GO TO 10
           ELSE

          if ((aa1.LE.100).and.(aa1.GT.10)) then
          GO TO 212
          ELSE

          if ((aa1.LE.1000).and.(aa1.GT.100)) then
          GO TO 30
          ELSE

          if ((aa1.LE.1e4).and.(aa1.GT.1e3)) then
          GO TO 40
          ELSE

!	if ((aa1.LE.1e5).and.(aa1.GT.1e4)) then
		if ((aa1.GT.1e4)) then

          GO TO 40
          ELSE

          GO TO 1
	end if
          END IF
          END IF
          END IF
          END IF
          END IF

1         X=1.
10       i5=0.78
         i6=2.*pi
         k81=100
         k82=30
         k822=30
         go to 104

212      i5=1.4
         i6=2.*pi
         k81=100
         k82=30
         k822=26
         go to 104

30       i5=1.5
         i6=2.*pi
         k81=300
         k82=60
         k822=26
         go to 104

40       i5=1.54
         i6=2.*pi
         k81=500
         k82=60
         k822=26
         go to 104



103     if (aa3.LE.10) then
        GO TO 50
        ELSE
        if ((aa3.LE.100).and.(aa3.GT.10)) then
        GO TO 60
        ELSE

        if ((aa3.LE.1e3).and.(aa3.GT.100)) then
        GO TO 70
        ELSE
         if ((aa3.LE.1e4).and.(aa3.GT.1000)) then
        GOTO 80
        ELSE
        GO TO 2
        END IF
        END IF
        END IF
        END IF
        
2       X=1.

50      i5=0.11
        i6=2.*pi
        k81=200
        k82=100
        k822=60
        go to 104

60      i5=0.05
        i6=2.*pi
        k81=300
        k82=100
        k822=60
        go to 104

70      i5=0.005
        i6=2.*pi
        k81=300
        k82=100
        k822=60
        goto 104
        
80      i5=0.0005
        i6=2.*pi
        k81=300
        k82=100
        k822=60
        goto 104      

104     I25 = I5 / 2.
        I26 = I6 / 2.

       
!****** 'TWO DIFFERENT AXES (1 aspect ratio)*************
    
         return
         END  
!*******************************************************