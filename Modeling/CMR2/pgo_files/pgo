#!/bin/sh
# starts N instances of a job
# ex.: pgo jubba.py 8
#      will start eight instances of python script jubba.py

function usage
{
  echo "Usage: pgo pyfile.py [num [-nb]"
  echo "           num - Number to start"
  echo "           -nb - No backup of script"
  exit 1
}

# check number of args
if [ $# -gt 3 ]; then
    usage
    exit 1
fi
if [ $# -lt 1 ]; then
    usage
    exit 1
fi



# look for runmfile in current dir
SH_PATH=`dirname $0`
if [ "$SH_PATH" = "" ]; then
    SH_PATH=`dirname \`which $0\``
fi
SH_SCRIPT=${SH_PATH}/runpyfile.sh

BACK_DIR=runs
PYFILE=$1
PNAME=`expr substr \`basename $PYFILE\` 1 15`
PROCS=$2
if [ $# -lt 2 ]; then
    PROCS=1
fi


# save history
if [ "$3" != "-nb" ]; then
    echo Making backup:
    mkdir -p $BACK_DIR
    echo cp $PYFILE runs/${PYFILE}.`date +%y%b%d_%H%M`
    cp $PYFILE runs/${PYFILE}.`date +%y%b%d_%H%M`
fi


# start scripts
echo Starting $PROCS script\(s\):
for i in `seq 1 $PROCS`; do
	echo $i
	#$2 1>/dev/null 2>/dev/null &
	echo qsub -l h_vmem=2G -cwd -N $PNAME -v PYFILE=$PYFILE $SH_SCRIPT; 
	qsub -l h_vmem=2G -N $PNAME -v PYFILE=$PYFILE $SH_SCRIPT; 
	sleep 2;
done


