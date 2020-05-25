Running model 1, model 2, and alignment scripts instructions:

Model 1 instructions: 

python model_1.py will run IBM model 1. Please use the following arguments to run the model correctly.
"--english_file_name", a pointer to the to the English sentences file.
"--french_file_name", a pointer to the to the french sentences file.
"--size", set the size of the corpus that we want to read. size is float and it describes the percent amount of data to read.
"--output_parameter_file_name", The parameter file to be generated by the end of training, it only saves the last iteration.
"--number_of_iterations" the number of iterations to run
"--verbose" whether you want to see output throughout the training, i.e. the current iteration number. 
"--output_parameters_every_epoch" whether you want to store the parameters after every iteration
"--enhanced_smoothing" use the smoothing as it is described in this paper: "Improving IBM Word-Alignment Model 1"

Here is the command I'm using to run model 1. 
python model_1.py --english_file_name ../data/hansards.e
 --french_file_name ../data/hansards.f --size 85 --output_parameter_file_name "output.json" --number_of_iterations 10 
 --verbose True




Model 2 instructions:

python model_2.py runs the IBM model 2. 
please use the above same arguments when running this script, just for a single exception.
Model_2 as an option to initialize the alignments from model_1 output.
To use this please specify the argument: --initialization_file towards a output from get_alignments.py txt file, 
which was produced by a model 1 parameters. 



Alignment instructions:

To get the alignment file use the python script called get_alignments.py 

python get_alignments.py gets 5 arguments: 
First argument is the english file txt
second argument is the french file txt 
Third argument is the parameters file of either model 1 or model 2. the get alignments handles both parameters file.
Fourth argument is the output alignment file.
Fifth argument, verbose: when this is equalt the "true" string,
it outputs an periodic output to screen to see progress, also outputs to the same file an intermediate result after 38 sentences.

This is an example how to run the get_alignments.py:
python get_alignments.py data/english.txt data/french.txt parameters_output.json my_alignment.txt false 




Double alignment instructions:

I also implemented the double alignment file to which get the model prametrs of e-to-f and f-to-e and outputs 
a alignment file. 
To get this alignment file use the python script called double_parameters_get_alignments.py 

python double_parameters_get_alignments.py gets 6 arguments: 
First argument is the english file txt
second argument is the french file txt 
Third argument is the "e-to-f" (normal) parameters file of model 1. the alignment scripts handles only model 1 parameter file.
Third argument is the "f-to-e" (reverse) parameters file of model 1.
Fifth  argument is the output alignment file.
sixth argument, verbose: when this is equal the "true" string,
it outputs an periodic output to screen to see progress, also outputs to the same file an intermediate result after 38 sentences.

This is an example how to run the get_alignments.py:
python double_parameters_get_alignments.py data/english.txt data/french.txt parameters_output.json reverse_parameters_output.json my_alignment.txt false 

