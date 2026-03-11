#!/usr/bin/env perl
##############################################################################
# % cd ~/docs/KyotoU-NTT-postdoc/SharedTask/rte/nict-annotation/blog_evalobj
# % cat *.tsv | $0 ~/docs/KyotoU-NTT-postdoc/SharedTask/browser/out-html/
#
# NICT評判アノテーション結果 *.tsv を次の形式に変換。
#  SID --> 評価保持者, 評価表現, 評価表現文字位置, 評価タイプ, 評価対象
##############################################################################

use strict;
use warnings;
use Storable qw(store);;

use utf8;
binmode STDOUT, ":encoding(utf8)";
binmode STDERR, ":encoding(utf8)";
binmode STDIN, ":encoding(utf8)";

my %all;
while(<STDIN>){
    chomp;

    (my $sid, my $sent, my $evaluator, 
     my $eval_exp, my $eval_type, my $evaluated) = split /\t/;

    next unless $evaluator && $eval_exp && $eval_type && $evaluated;

    my @evaluator_a = split /\\n/, $evaluator;
    my @eval_exp_a =  split /\\n/, $eval_exp;
    my @eval_exp_position_a = @{ &get_eval_exp_position($sent, \@eval_exp_a, $_) };
    my @eval_type_a =  split /\\n/, $eval_type;
    my @evaluated_a =  split /\\n/, $evaluated;

    for my $i (0 .. $#evaluator_a){
	push @{$all{$sid}}, [ $evaluator_a[$i], $eval_exp_a[$i],
			      $eval_exp_position_a[$i],
			      $eval_type_a[$i],
			      $evaluated_a[$i] ];
    }
}

my $outfile = "$ARGV[0]/nict_sentiment_annotation.bin";
store(\%all, $outfile) or die "Can't write into $outfile: $!";

sub get_eval_exp_position {
    (my $sent, my $a_ref, my $line) = @_;
    my @eval_exp_position_a;
    for my $eval_exp (@$a_ref){
	push @eval_exp_position_a, &get_start_end_position($sent, $eval_exp, $line);
    }
    return \@eval_exp_position_a;
}

sub get_start_end_position {
    (my $sent, my $exp, my $line) = @_;
    my $start = index($sent, $exp);
    my $length = length $exp;
    my $end = $start + --$length;

    if($start == -1 || $end == -1){
	die "$line, SENT:[$sent], EXP:[$exp]";
    }

    return $start.'-'.$end;
}
