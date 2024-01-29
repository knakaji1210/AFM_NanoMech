import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#originallsbWave = np.zeros(16384)
number10=[]
#ヘッダー情報のパラメータ
zRange_V=1.586909
zSens_nm_V=8.500600

#ファイルを開く
with open(r'heikou.000', "rb")as f:
#ファイルの読み込み場所を指定
    f.seek(40960)
    data=f.read(32768)
#16進数に変換
    data = data.hex()
#2byte=1pixelなので4文字ずつスライスする。    #print(data)
    datahex= [data[i: i + 4] for i in range(0, len(data), 4)]

    #print(datahex)


#符号付き16進数として処理をし、10進数にする関数
def twosComplement_hex(hexval):
   bits = 16
   val = int(hexval, bits)
   if val & (1 << (bits - 1)):
      val -= 1 << bits
   return val

# print(datahex)
#ビックエンディアンで10進数に変換する。
for i in datahex:
    bytes_be = bytes.fromhex(i)
    bytes_le = bytes_be[::-1]
    hex_le = bytes_le.hex()
    #print(hex_le)
    data10=twosComplement_hex(hex_le)
    number10.append(data10)
#print(number10)

#print(len(number10))
#print(number10)
#行列を作成する
originallsbWave = np.array(number10)
#print(originallsbWave)
originallsbWave=(((originallsbWave/65536)*zRange_V*zSens_nm_V)/(10**9))
#行列を成形する
originallsbWave=np.reshape(originallsbWave,(128,128))
#表示画像に合わせるため、転置と回転を行う。
originallsbWave=originallsbWave.T
#print(originallsbWave)
originallsbWave=np.rot90(originallsbWave,k=1)
#print(originallsbWave)
plt.figure()
sns.heatmap(originallsbWave)
plt.show()


#print(originallsbWave)

