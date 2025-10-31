LOGFILE="output.log"
CONF="-b --full_config json://configuration.json"
#CONF="-b --configuration json://configuration_nonmc.json"
OUTPUT_DIR="--aod-writer-json output_director.json"
INPUT="--aod-file @input_data.txt"

o2-analysis-lf-cluster-studies-tree-creator $CONF |
    
    o2-analysis-mccollision-converter $CONF |
    
    o2-analysis-propagationservice $CONF |
    o2-analysis-trackselection $CONF |
    o2-analysis-tracks-extra-v002-converter $CONF |
    o2-analysis-event-selection-service $CONF |
    o2-analysis-pid-tof-merge $CONF |
    o2-analysis-pid-tpc-service $CONF |

    o2-analysis-ft0-corrected-table $CONF --aod-file @input_data.txt --aod-writer-json output_director.json > $LOGFILE

# report the status of the workflow
rc=$?
if [ $rc -eq 0 ]; then
    echo "Workflow finished successfully"
else
    echo "Error: Workflow failed with status $rc"
    echo "Check the log file for more details: $LOGFILE"
    # exit $rc
fi