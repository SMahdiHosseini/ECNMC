# The script generates new .dat trace files starting from the .pcap files. In the first step only packets which are (>=20bytes) are extracted from the .pcap files. In the second step double ips are replaced by only the first one of them. In the third step the 4 parts of the ip addresses are stored into 4 different columns in the tab-delimited (.tbdlm) .dat file. Finally in the last step all traces have to be imported to MATLAB and each one can be saved to a .mat through the importTracesToMATLAB script.  

extract (){
for f in $@
do
    echo "tshark $f ... and store to a tab delimited .dat file"

    time tshark -r $f -Y "tcp" -T fields -e frame.time_relative -e ip.src -e tcp.srcport -e ip.dst -e tcp.dstport -e frame.len -E separator=',' > $f.csv &  #   Filter pcap file for only IPv4 traffic and store packet details in different fields
    # sed -e 's/\,[.1234567890]*\t*// g' |  #   Sanitize double ips in $f and keep only the first one
    # awk '$1=$1' FS="." OFS="." > $f.csv &  #   Add , delimiters and save to file...

done
wait
}



PATH_TO_DATA="equinix-chicago.dirA.20090820-125904.UTC.anon.pcap"


extract $PATH_TO_DATA

printf "done \n"
