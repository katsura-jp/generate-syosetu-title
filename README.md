# なろう風タイトルジェネレータ

## 環境構築
```
git clone ...
cd generate-naro-title
pip install -r requirements.txt
mkdir model
```

学習済みモデルは(Google Drive)[https://drive.google.com/drive/folders/1mMo74ytvXJfpInCpovllPhDevXA7vjYQ?usp=sharing]にアップロードしているのでダウンロードして使ってください（バージョンは1つでも全部でも）. ダウンロードしたzipファイルを作った`model`ディレクトリに移動させて解凍してください．

`model`ディレクトリは任意の名前で問題ないので，嫌な人は別名でどうぞ．さらに言えば，学習済みモデルへのパスは任意に指定できるのでダウンロード先で解凍してそこを指定してもOKです．

## 使い方
（例）`model`にversion_1を入れ，`input.txt`にあらすじを記述した場合．また，ビームサーチのサイズに5を指定．

```
python 
```


