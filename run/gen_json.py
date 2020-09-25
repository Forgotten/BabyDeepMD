# Open function to open the file "MyFile1.txt"  
# (same directory) in append mode and 

Nsamples = [100]

alpha1 = 0.5
alpha2 = 0.5

exec_file = "2BodyForcesDist2D_Periodic_NUFFT_mixed.py"
for nsample in Nsamples: 

    # we create the json file
    nameFile = "mixed_2D_alpha1_%.1f_alpha2_%.1f_Ns_%d.json"%((alpha1, alpha2,nsample))
    filew = open(nameFile,"w") 
    filer = open("basic_json.txt","r") 

    filew.write("{ \n")

    # adding all the intermediate values 
    for line in filer.readlines():
        filew.write(line)


    filew.write("\"Nsamples\":          %d,\n"%(nsample))   
    filew.write("\"weight1\":           %.1f,\n"%(alpha1))
    filew.write("\"weight2\":           %.1f \n"%(alpha2))    

    filew.write("}")

    filew.close()
    filer.close()

    filesh = open("slurm_mixed_2D_alpha1_%.1f_alpha2_%.1f_Ns_%d.sh"%((alpha1,\
                                                                      alpha2,\
                                                                      nsample)), "w")
    filersh = open("basic_sh.txt","r") 

        # adding all the intermediate values 
    for line in filersh.readlines():
        filesh.write(line)

    filesh.write(exec_file + " " + nameFile)

    filesh.close()
    filersh.close()

    # we create the sh file 