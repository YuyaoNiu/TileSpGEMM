import matplotlib.pyplot as plt 
import numpy as np 

#FullChip
#times1=[0.46, 0.89, 9.25, 2270.76]
#times2=[0.42, 0.60, 1.19, 165.41]
#times3=[0.38, 0.54, 0.68, 0.83]

#kkt_power
gflops=[0.24404, 0.48150,0.93040 ,1.45555,1.66449]
#times2=[0.38, 0.59, 1.03, 152.47]
#times3=[0.28, 0.40, 0.48, 0.49]
X=np.arange(len(gflops))
bar_width=0.25
tick_label = ['1','2','4','8','16']


ax=plt.figure(figsize=(5,4))
#fig=plt.gcf()
#fig.set_size_inches(185,105)
plt.bar(X, gflops, bar_width, align="center", color="royalblue", label="Column block")
#plt.bar(X+bar_width, times2, bar_width, color="lightskyblue", align="center", label="Row block")

#plt.bar(X+bar_width+bar_width, times3, bar_width, color="lightgreen", align="center", label="Recursive block")

font1 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 14,
}

font2 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 11,
}



plt.xlabel('Threads',font1)
plt.ylabel('GFlops ',font1) 

plt.xticks(X, tick_label)


plt.grid(axis="y",linestyle='--',alpha=0.6)
#FullChip
#plt.text(2.02, 1.73,'9.25', rotation=90, wrap=True,c='black',fontsize=9)
#plt.text(3.02, 1.50,'2270.76', rotation=90, wrap=True,c='black',fontsize=9)
#plt.text(3.02+bar_width, 1.60,'165.41', rotation=90, wrap=True,c='black',fontsize=9)
#kkt_power
#plt.text(2.02, 1.73,'8.23', rotation=90, wrap=True,c='black',fontsize=9)
#plt.text(3.02, 1.52,'1505.75', rotation=90, wrap=True,c='black',fontsize=9)
#plt.text(3.02+bar_width, 1.60,'152.47', rotation=90, wrap=True,c='black',fontsize=9)
#plt.text(3.02+2*bar_width, 1.73,'126.59', rotation=90, wrap=True,c='black',fontsize=9)



plt.ylim(0,2)
#plt.legend(loc="upper left",prop=font2,fancybox=False)
#plt.savefig('combench-parabolic_fem.png',dpi=100)
#plt.savefig('combench-ASIC_680k.eps',dpi=100)
plt.show() 
