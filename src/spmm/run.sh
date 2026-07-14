#!/bin/bash
function ergodic_run() {
targetDir=`ls $1`

  for file in $targetDir
    do
      if [ -d  $1"/"$file ]
      then
        ergodic_run  $1"/"$file
      else
        if [ ${file:0-4} == '.mtx' ]  ;
          then
            make run input_matrix="none" test_matrix=$1"/"$file log_path="/data/seery/src/log/reslut/3_251113.log";
      fi
  fi
  done

}

path_str=/data/seery/dataset/dlmc/rn50
ergodic_run $path_str