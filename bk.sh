#!/bin/bash

if [ -z $1 ]||[ -z $2 ]; then
	echo usage: $0 [sourcedir] [targetdir]
else
	SRCDIR=$1
	TRGDIR=$2
	BKFILE=backup.$(date +%y%m%d%H%M%S).tar.gz
	if [ -d $TRGDIR ]; then
		tar -cvzf - $SRCDIR | split -b 25m - $TRGDIR/$BKFILE
	else
		mkdir $TRGDIR
		tar -cvzf - $SRCDIR | split -b 25m - $TRGDIR/$BKFILE
	fi
fi

