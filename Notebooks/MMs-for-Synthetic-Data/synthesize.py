import random
import sys
import pandas as pd

if (len(sys.argv)==1):
	print("Usage:")
	print(sys.argv[0]+" <csvfile> [<numsamples>]")
	sys.exit(0)

df=pd.read_csv(sys.argv[1])

numsamples=df.shape[0]

try:
	numsamples=int (sys.argv[2])
except:
	pass

columns=[]
for column in df:
	columns=columns+[column]

for i in range(0,len(columns)-2):
	print(columns[i], end=",")
print(columns[i+1])

mins=[]
maxs=[]

for i in range(0,len(columns)):
	mins=mins+[df[columns[i]].min()]
	maxs=maxs+[df[columns[i]].max()]

for i in range(0,numsamples):
	for j in range(0,len(columns)-2):
		out=random.uniform(mins[j], maxs[j])
		if ("int" in str(df.dtypes.iloc[j])):
			out=int(out)
		print(out, end=",")

	out=random.uniform(mins[j+1], maxs[j+1])
	if ("int" in str(df.dtypes.iloc[j+1])):
		out=int(out)
	print(out)


