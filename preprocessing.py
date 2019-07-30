from glob import glob
import cv2
import os
import subprocess
import random
from shutil import copyfile

folder = "/home/users/datasets/UTFPR-CMMD/Etapa2-Crop/"

def rename(folder):	
	count=1
	make = os.listdir(folder)
	for mk in sorted(make):
		model = os.listdir(folder+mk)
		for md in sorted(model):
			path = str(folder)+mk+"/"+md
			files = os.listdir(path)
			for f in sorted(files):
				src = folder+mk+"/"+md+"/"+f
				dst = str(path)+"/"+str(mk)+"_"+str(md)+"_"+str(count).zfill(3)+".jpg"
				print dst
				#src = os.path.join(path,f)
				#dst = os.path.join(path,str(count) + ".jpg")
				os.rename(src,dst)							
				count +=1
				#print "jpg"
			count=1


def clean(folder):	
	make = os.listdir(folder)
	count = 0
	arquivo = open('log.txt','w')
	for mk in sorted(make):
		model = os.listdir(folder+mk)
		for md in sorted(model):
			path = str(folder)+mk+"/"+md
			files = os.listdir(path)
			print md
			for f in sorted(files):
				try:
					path = folder+mk+"/"+md+"/"+f
					#print (path)
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

def cp(folder):	
	from shutil import copyfile
	count=1
	make = os.listdir(folder)
	for mk in sorted(make):
		model = os.listdir(folder+mk)
		for md in sorted(model):
			path = str(folder)+mk+"/"+md
			files = os.listdir(path)
			selected_files = random.sample(files,k=5)
			print (selected_files)
			for f in selected_files:
				src = folder+mk+"/"+md+"/"+f
				dst = "/home/users/datasets/UTFPR-SCD/Original2/"+f
				copyfile(src,dst)		# BUUUUU LUUUU :P					
				count +=1
				#print "jpg"
			count=1

folder2 = "/home/users/datasets/UTFPR-SCD/Original/"
def rmp(folder):	
	from shutil import copyfile
	count=0
	arq=open("out2.txt","w")
	make = os.listdir(folder)
	for mk in sorted(make):
		model = os.listdir(folder+mk)
		for md in sorted(model):
			files = os.listdir(folder+mk+"/"+md)
			for f in files:
				img = cv2.imread(folder+mk+"/"+md+"/"+f)
				imh = img.shape[0]
				imw = img.shape[1]
				if imw < 150:
					count +=1
					log = str(f)+"-"+str(imw)
					print log
					arq.write(log)
					arq.write("\n")
					#os.remove(folder+mk+"/"+md+"/"+f)
	print count
	arq.write(count)
	arq.close()

def rename_scd(folder2):	
	from shutil import copyfile
	count=1
	files = os.listdir(folder2)
	for f in files:
		src = folder2+f
		dst = "/home/users/datasets/UTFPR-SCD/Original2/"+str(count).zfill(3)+".jpg"
		copyfile(src,dst)		# BUUUUU LUUUU :P	
		count+=1

#rename(folder)
#rename_scd(folder2)
rmp(folder)
