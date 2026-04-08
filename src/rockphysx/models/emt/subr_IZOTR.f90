!*********************************

	subroutine izotr(c,c_iz)
		 		USE MSFLIB

!* Chaotically rotated inclusions in sample volume
		implicit real*8(a-h,o-z)
	dimension c(3,3,3,3),c_iz(3,3,3,3)
	common/pi/pi
		common/error/ind_err
	INTEGER(2) control, newcontrol,status

	CALL GETCONTROLFPQQ(control)
	newcontrol = (control .OR. FPCW$zerodivide.or.fpcw$invalid.or. fpcw$overflow)
	! Invalid exception set (disabled).
	CALL SETCONTROLFPQQ(newcontrol)


	do i=1,3
	do j=1,3
	do k=1,3
	do l=1,3
	c_iz(i,j,k,l)=0.
	end do
	end do
	end do
	end do

	c11=c(1,1,1,1)
	c22=c(2,2,2,2)
	c33=c(3,3,3,3)

	c12=c(1,1,2,2)
	c13=c(1,1,3,3)
	c23=c(2,2,3,3)

	c44=c(2,3,2,3)
	c55=c(1,3,1,3)
	c66=c(1,2,1,2)

!	fortran(ir1111,optimized);
	t1 = 0.3141593E1**2
	t20 = 32.E0/15.E0*c44*t1+8.E0/5.E0*c22*t1+16.E0/15.E0*c23*t1+32.E0&
     /15.E0*c55*t1+8.E0/5.E0*c11*t1+32.E0/15.E0*c66*t1+16.E0/15.E0*c13*&
     t1+16.E0/15.E0*c12*t1+8.E0/5.E0*c33*t1

	 c_iz(1,1,1,1)=t20/8/pi/pi

! fortran(ir1222,optimized);
      t1 = 0.3141593E1**2
	t20 = 32.E0/15.E0*c12*t1+8.E0/15.E0*c11*t1+8.E0/15.E0*c22*t1-16.E0&
     /15.E0*c66*t1+32.E0/15.E0*c13*t1+32.E0/15.E0*c23*t1-16.E0/15.E0*c44*t1+8.E0/15.E0*c33*t1-16.E0/15.E0*c55*t1


	c_iz(1,1,2,2)=t20/8/pi/pi
	c_iz(2,2,1,1)=c_iz(1,1,2,2)
! fortran(ir4444,optimized);
      t1 = 0.3141593E1**2
	t20 = -8.E0/15.E0*c12*t1+8.E0/15.E0*c22*t1+8.E0/5.E0*c44*t1+8.E0/5.E0*c55*t1+8.E0/15.E0*c11*t1+8.E0/5.E0*c66*t1-8.E0/15.E0*c13*t1-8.E0/15.E0*c23*t1+8.E0/15.E0*c33*t1

	c_iz(1,3,1,3)=t20/8/pi/pi
	 
!***************************************
	c_iz(2,2,2,2)=c_iz(1,1,1,1)
	c_iz(3,3,3,3)=c_iz(1,1,1,1)

	c_iz(1,1,3,3)=c_iz(1,1,2,2)
	c_iz(2,2,3,3)=c_iz(1,1,2,2)

		c_iz(3,3,1,1)=c_iz(1,1,2,2)
		c_iz(3,3,2,2)=c_iz(1,1,2,2)

	c_iz(1,3,3,1)=c_iz(1,3,1,3)
	c_iz(3,1,3,1)=c_iz(1,3,1,3)
	c_iz(3,1,1,3)=c_iz(1,3,1,3)

	c_iz(1,2,1,2)=c_iz(1,3,1,3)
	c_iz(1,2,2,1)=c_iz(1,3,1,3)
	c_iz(2,1,1,2)= c_iz(1,3,1,3)
	c_iz(2,1,2,1)= c_iz(1,3,1,3)

	c_iz(2,3,2,3)=C_iz(1,3,1,3)
	c_iz(2,3,3,2)=c_iz(1,3,1,3)
	C_IZ(3,2,2,3)=c_iz(1,3,1,3)
	C_iz(3,2,3,2)=c_iz(1,3,1,3)

CALL GETSTATUSFPQQ(status)
	IF ((status .AND. FPSW$zerodivide) > 0) THEN
!	WRITE (*, *) "Zero divide has occurred"
	 ind_err=12
	end if


		CALL GETSTATUSFPQQ(status)
	IF ((status .AND. FPSW$invalid) > 0) THEN
!	WRITE (*, *) "Invalid number
	 ind_err=124
	end if

			CALL GETSTATUSFPQQ(status)
	IF ((status .AND. FPSW$overflow) > 0) THEN
!	WRITE (*, *) "Overflow"
      ind_err=127
	end if

	return
	end

!****************************************
	