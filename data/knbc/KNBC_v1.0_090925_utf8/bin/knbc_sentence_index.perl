#!/usr/bin/env perl
##############################################################
# % $0 KN001_Keitai_1/ out-html/
#
# 1記事中の全文のインデックスページの作成。
# KN001_Keitai_1/ は KN001_Keitai_1.tar.gz を解凍したもの。
# out-htmlは出力先ディレクトリ。
##############################################################

use strict;
use warnings;
use utf8;
use File::Basename;

binmode STDOUT, ":encoding(utf8)";
binmode STDERR, ":encoding(utf8)";
binmode STDIN, ":encoding(utf8)";

my $article = basename $ARGV[0];
$article =~ s/\/+$//;

my @sentences = sort {
    my $tmpa = (split /-/, $a)[2];
    unless(defined $tmpa){
	die "DIE-a:$a";
    }
    my $tmpb = (split /-/, $b)[2];
    unless(defined $tmpb){
	die "DIE-b:$b";
    }

    (split /-/, $a)[2] <=> (split /-/, $b)[2]
} glob("$ARGV[0]/KN*");

my $out = "$ARGV[1]/$article.html";

open OUT, ">:encoding(utf8)", $out or die "Can't write into $out: $!";

print OUT "<html><head><title>$article</title><meta http-equiv=\"content-type\" content=\"application/html; charset=UTF-8\" /></head>";
print OUT "<link rel=\"stylesheet\" type=\"text/css\" href=\"knbc_sentence_index.css\" />";
print OUT "<body><table>";

for my $sent (@sentences){
    open IN, "<:encoding(utf8)", $sent or die "Can't open $sent: $!";
    my $tmp_str;
    while(<IN>){
	next if /^[*+]/;
	chomp;
	if(/^\#/){
	    print OUT "<tr><td id=sid>";
	    print OUT $_;
	    print OUT "</td></tr>";
	}elsif($_ eq 'EOS'){
	    print OUT "<tr><td id=sent>";
	    my $sent_id = basename $sent;
	    print OUT "<a href=\"$sent_id.html\" target=\"_blank\">●</a> ";
	    print OUT $tmp_str;
	    print OUT "</td></tr>";
	    $tmp_str = undef;
	}else{
	    $tmp_str .= (split / /)[0];
	}
    }
    close IN;
}

print OUT "</table></body></html>";
