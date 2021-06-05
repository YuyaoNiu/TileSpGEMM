import numpy as np
import pandas as pd
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


#data=pd.read_csv('spmv.csv')
data=pd.read_csv('blocksize-offshore-32*32.mtx.csv')
#data=pd.read_csv('1111.csv')

df=pd.DataFrame(data)
data=df.pivot_table("task","row","col")
#data=df.pivot_table("methods","Empty ratio(%)","nnz/row")

#list=data.values.tolist()



#print(list)
font = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 13,}
font1 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 15,}


f,ax = plt.subplots()
#plt.figure(facecolor='snow')
#sns.heatmap(data, annot=True,fmt='.20g',cmap='Blues', cbar=False,annot_kws={'size':12,} )
sns.heatmap(data, annot=False,fmt='.20g',cmap='Blues', cbar=False,annot_kws={'size':9,} )



#ax.imshow(list,cmap="summer_r",origin="lower")

#ax.set_xlabel('row',font1)

#ax.set_ylabel('col',font1)

#ax.set_ylabel('emptyratio(%)',font1)
#ax.set_ylabel('nlevels(%)',font)

#ax1=plt.gca()

#ax1.patch.set_facecolor("whitesmoke")    


#ax.xaxis.set_major_locator(x_major_locator)

#plt.hlines(2900, 0, 32, colors='darkblue',)
#plt.hlines(100, 0, 1.1, colors='darkblue',)
#plt.vlines(1.1, 15, 100, colors='darkblue',)
#plt.hlines(15, 1.1, 20, colors='darkblue',)
#plt.vlines(20, 0, 15, colors='darkblue',)

#plt.text(1, 3500, "cuSPARSE", font)
#plt.text(1, 2500, "Sync-free", font)



#plt.hlines(10, 1, 7.9, colors='darkblue',)
#plt.vlines(7.9, 0, 10, colors='darkblue',)
#plt.hlines(98, 0, 1, colors='darkblue',)
#plt.vlines(1, 10, 98, colors='darkblue',)

#plt.text(1, 480, "vector-syncfree", font)
#plt.text(3, 60, "scalar-levelset", font)

#plt.hlines(256, 0, 12, colors='darkblue',)
#plt.vlines(12, 0, 512, colors='darkblue',)
#plt.hlines(70, 12, 32, colors='darkblue',)


#plt.text(0, 20, "scalar-CSR", font)
#plt.text(0, 480, "scalar-DCSR", font)
#plt.text(12.5, 20, "vector-CSR", font)
#plt.text(12.5, 480, "vector-DCSR", font)



#plt.xlim(0,17)
#plt.ylim(0,250)
#plt.ylim(0,20)
#axins = ax.inset_axes([0.3, 0.3, 0.27, 0.27])
#sns.heatmap(data2,cmap="summer_r",cbar=False)


# sub region of the original image
#x1, x2, y1, y2 = 0, 17, 0, 15
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
# fix the number of ticks on the inset axes
#axins.yaxis.get_major_locator().set_params(nbins=10)
#axins.xaxis.get_major_locator().set_params(nbins=10)

#plt.xticks(visible=True)
#plt.yticks(visible=True)
#plt.savefig('spmv.pdf',dpi=300)
#plt.savefig('sptrsv.pdf',dpi=300)
plt.savefig('offshore-32*32.mtx.pdf',dpi=300)
plt.show()

