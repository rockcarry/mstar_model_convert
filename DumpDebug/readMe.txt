gcc code/splite_binary_to_binary.c -o ./splite_binary_to_binary
gcc code/splite_binary_to_string.c -o ./splite_binary_to_string
gcc code/splite_dump_binary -o ./splite_dump_binary
gcc code/compareAccuracy.c -o ./compareAccuracy -lm

	*****************
	compare accuracy
	*****************
	cd IPUSDK/Tool/DumpDebug
	(cd IPUSDK/DumpDebug/  if In Release SDK )
	./auto_dump_debug.sh ../ sample.bin benchmark.bin
	e.g:
		./auto_dump_debug.sh ../ ~/tmp/tiny_float.bin ~/tmp/tiny_fixed.bin
	
	**************************************
	change binary to string for one tensor
	**************************************
	cd IPUSDK/Tool/DumpDebug
	(cd IPUSDK/DumpDebug/  if In Release SDK )
	./binary_to_string.sh binary_file_dir
	e.g:
		./binary_to_string.sh ./Undefined_tiny_cmodel_fixed.bin_DumpDebug_out/genCompareLayer/0.156.xx.output0/

	***************************************
	splite binary to string for all tensors
        ***************************************
	./splite_dump_binary binary_file
	e.g:
		./splite_dump_binary ~/tmp/tiny_float.bin

ps: All cmds  must run in DumpDebug directory
