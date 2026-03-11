#!/usr/bin/env perl
############################################################
# % $0 SYNSEM_ANN/ out-html/
#
# KNBC全記事のインデックスページを作成。
# out-htmlは出力先ディレクトリ。
############################################################

use strict;
use warnings;
use utf8;
use File::Basename;

binmode STDOUT, ":encoding(utf8)";
binmode STDERR, ":encoding(utf8)";
binmode STDIN, ":encoding(utf8)";

my %categories;

my @articles = glob("$ARGV[0]/KN*");

for my $article (sort @articles){
    my $article = basename $article;
    my $heading = &get_heading($article);
    if($article =~ /Keitai/){
	push @{$categories{keitai}}, [ $article, $heading ];
    }elsif($article =~ /Kyoto/){
	push @{$categories{kyoto}}, [ $article, $heading ];
    }elsif($article =~ /Gourmet/){
	push @{$categories{gourmet}}, [ $article, $heading ];
    }elsif($article =~ /Sports/){
	push @{$categories{sports}}, [ $article, $heading ];
    }else{
	die "Unknown article category: $article";
    }
}

### 出力

my @title = qw(携帯電話 京都観光 グルメ スポーツ);
my @category = qw(keitai kyoto gourmet sports);

for my $n (0 .. $#category){

    print STDERR "$ARGV[1]/$category[$n].html\n";

    my $html = "$ARGV[1]/$category[$n].html";
    open OUT, ">:encoding(utf8)", $html or die "Can't open $html: $!";
    
    print OUT "<html><head><title>$title[$n]</title><meta http-equiv=\"content-type\" content=\"application/html; charset=UTF-8\" /></head>";
    print OUT "<link rel=\"stylesheet\" type=\"text/css\" href=\"knbc_article_index.css\" />";

    print OUT "<table><tr>";
    for my $m (0 .. $#category){
	print OUT "<th id=articleindexheading>";
	if($n ne $m){
	    print OUT "<a href=$category[$m].html>$title[$m]</a>";
	}else{
	    print OUT "$title[$m]";
	}
	print OUT "</th>";
    }
    print OUT "</tr></table>";

    print OUT "<table>";
    for my $art (@{$categories{$category[$n]}}){
	print OUT "<tr><td id=\"artline\">";
	print OUT "<a href=\"${$art}[0].html\">${$art}[0]</a>&nbsp;&nbsp;";
	print OUT "<small>（<a href=\"${$art}[0]-annotation.html\">全文全解析</a>）</small>&nbsp;&nbsp;";
	print OUT "${$art}[1]";
	print OUT "</td></tr>";
    }
    print OUT "</table>";

    print OUT "</html>";
    close OUT;
}


sub get_heading {
    my $article = shift;
    my $filename = "$ARGV[0]/$article/$article-1-1-01";
    my $result = '';
    open ART, "<:encoding(utf8)", $filename or die "Can't open $filename: $!";
    while(<ART>){
	next if /^[\#*+E]/;
	$result .= (split / /)[0];
    }
    close ART;
    return $result;
}
