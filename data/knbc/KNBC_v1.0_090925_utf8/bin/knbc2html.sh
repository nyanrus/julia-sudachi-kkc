#!/bin/sh
###############################################################
# % $0 out-html/
#
# KNBCをHTML化するスクリプト。
# out-htmlは出力先ディレクトリ。
###############################################################

# コーパス本体1の場所
SYNSEM_ANN=../corpus1/
# コーパス本体2の場所
EVAL_ANN=../corpus2/

# 評判アノテーションのハッシュ化
echo "Reading in the Evaluation Annotation...";
cat $EVAL_ANN/*.tsv | ./nict_sentiment.perl $1

# 全記事のインデックスページの作成
echo "Making index pages of all articles...";
./knbc_article_index.perl $SYNSEM_ANN $1

# 1記事中の全文のインデックスページの作成
echo "Making index pages of all sentences for each article...";
for d in `find $SYNSEM_ANN -maxdepth 1 -mindepth 1 -type d`
do
 ./knbc_sentence_index.perl $d $1
done

# 1記事中の全文の係り受け・意味関係ページと形態素ページの作成
echo "Making annotation pages for each article...";
for d in `find $SYNSEM_ANN -maxdepth 1 -mindepth 1 -type d`
do
 ./knbc_annotation_per-article.perl $d $1
done

# 1文の係り受け・意味関係ページと形態素ページの作成
echo "Making annotation pages for each sentence...";
for f in `find $SYNSEM_ANN -maxdepth 2 -mindepth 2 -type f -name KN*-[0-9]*`
do
 ./knbc_annotation.perl $f $1
done

# index.html を keitai.html にリンク
PWD=`pwd`
ln -s $PWD/$1/keitai.html $PWD/$1/index.html

# スタイルシートのコピー
cp *.css $1/

