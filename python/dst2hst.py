# create a dst file
f=open("x.dst", "w")
for i in range(256):
    f.write(str(rnd.randrange(256))+'\n')
f.flush()
f.close()

# open the file in dst2hst. Can fix stderr later. 
f.open('x.dst', "r")
d2h=subprocess.Popen("dst2hst", stdout=subprocess.PIPE, stdin=f)
for line in d2h.stdout:
    print(line.rstrip().decode('ASCII'))

    
