from glob import glob
import cv2
import os
import subprocess

folder = "/home/users/datasets/UTFPR-CMMD/afazer/Ford/Del-Rey/"

def rename(folder):	
	count=1
	files = os.listdir(folder)
	for f in sorted(files):
		src = folder+f
		dst = str(folder)+"Ford"+"_"+"Del-Rey"+"_"+str(count).zfill(3)+".jpg"
		print dst
		#src = os.path.join(path,f)
		#dst = os.path.join(path,str(count) + ".jpg")
		os.rename(src,dst)							
		count +=1
		#print "jpg"
	count=1


def clean(folder):	
	count = 0
	arquivo = open('log2.txt','w')
	files = os.listdir(folder)
	for f in sorted(files):
		try:
			path = folder+f
			print (path)
			output = str(subprocess.check_output("file --extension "+path, shell=True))
			#print (output)			
			a = output.find("jpeg/jpg/jpe/jfif")
			#print (a)
			#print "----"
			if a == -1:
				print ("IMAGEM ----- "+f+" ----- DELETADA POR FORMATO INCORRETO")
				arquivo.write("IMAGEM ----- "+f+" ----- DELETADA POR FORMATO INCORRETO\n")			
				os.remove(path)
				count +=1
		except:
			print ("IMAGEM ----- "+f+" ----- DELETADA POR CORRIMPIMENTO")
			arquivo.write("IMAGEM ----- "+f+" ----- DELETADA POR CORROMPIMENTO\n")				
			os.remove(path)
			count +=1
	print (str(count)+" Imagens deletadas")
	arquivo.write(str(count)+" Imagens deletadas")

rename(folder)
clean(folder)
