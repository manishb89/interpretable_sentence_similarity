#!/usr/bin/perl
#
# based on: 
#   Melamed, I. Dan. 1998. 
#   Manual annotation of translational equivalence: The blinker
#   project. Technical Report 98-07, Institute for
#   Research in Cognitive Science, Philadelphia,
#
#   http://repository.upenn.edu/cgi/viewcontent.cgi?article=1055&context=ircs_reports
#
#
#  given: 
#
#    g = gold standard token:token alignments (produced aligning all
#        tokens in chunk:chunk alignments)
#
#    s = system token:token alignments (produced aligning all tokens
#        in chunk:chunk alignments)
#
#  we discard punctuation .,:'`?;"-
#
#  we define
#    precis(g,s) = | overlap(g,s) | / |s|                   (eq. 1 in Melamed 98)
#    recall(g,s) = | overlap(g,s) | / |g|                   (eq. 2 in Melamed 98)
#                                            (F1 is equal to eq. 3 in Melamed 98)
#
#  where overlap returns the number of token:token alignments in
#  common between both sets
#
#  g and s can be fuzzy sets, where each token:token alignment is
#  weighted as follows:
#
#    weight(t1:t2) = 1 / max(fanout(t1),fanout(t2))        (eq. 4 in Melamed 98)
#
#    given directed alignments, fanout(t) is the number of token:token
#    alignments which have their origin in t #
#
#  As we have a different fanout factor in the gold standard pair and
#  in the system pair we use the fanout of the sys to compute overlap
#  and |s| for precision, and the fanout of the gold standard to
#  compute overlap and |g| for recall.  #
#
#  Precision and recall are computed for all alignments of all pairs
#  in one go (i.e. as opposed to aeraging F1 of each sentence pair)
#
# The script provides four evaluation measures:
#
# - F1 where alignment type and score are ignored (F1A)
# - F1 where alignment types need to match, but scores are
#   ignored. Match is quantified using Jaccard, as there can be multiple
#   tags (FACT,POL).  (F1AT)
# - F1 where alignment type is ignored, but each alignment is penalized
#   when scores do not match (F1AS)
# - F1 where alignment types need to match, and each alignment is
#   penalized when scores do not match. Match is quantified using
#   Jaccard, as there can be multiple tags (FACT,POL). In addition the
#   following special cases are catered for:
#   . there is no type penalty between tags {SPE1, SPE2, REL, SIMI} when
#     scores are (0-2] 
#   . there is no type penalty between EQUI and SIMI/SPE with score 4
#     (F1AST) 

#
# When type needs to match, a token:token alignment is in the overlap
# iff the types of the alignment in system and gold standard files is
# the same.
#
# When the scores are taken into account, the weight of the
# token:token alignment is penalized for differenes in score between
# the system and gold-standard alignment, as follows: #
#
#    weight(t1:t2) = 1 / max(fanout(t1),fanout(t2))     (eq. 4 in Melamed 98)
#                  * ( 1 - abs(score(t1:t2,sys) - score(t1:t2)) / 5)
# 
#  Changes:
#  v1 Oct. 16 2014
#  - first release

#  v2 Nov.  7 2014
#  - bug fixed: Bug affecting alignments which had multiple types, as
#    equalset received tags concatenated by _
#  - changes to address the following special case for F1AST: 
#     . there is no type penalty between tags {SPE1, SPE2, REL, SIMI} 
#       when both scores are (0-2]
#     . there is no type penalty between EQUI and SIMI/SPE with score 4.

#  v2 Sep. 7 2015 (minor changes ~ inigo lopez-gazpio)
#  - Do not raise warnings when tokens are used in several distinct alignments (allow M:N alignments)


=head1 $0

=head1 SYNOPSIS

 evalF1.pl gs system --debug=[01]

 Outputs the F1

 Example:

   $ ./evalF1.pl gs sys --debug=1

 Author: Eneko Agirre
          
 Nov. 7, 2014


=cut


use Getopt::Long qw(:config auto_help); 
use Pod::Usage; 
use warnings;
use strict;
use List::Util qw(max) ;
use File::Basename;

my $DBG = 0 ;

GetOptions("debug=i" => \$DBG)
    or
    pod2usage() ;

pod2usage if $#ARGV != 1 ;

my $dir = dirname(__FILE__);
my $incorrect = system "perl $dir/wellformed.pl $ARGV[1] > /dev/null" ;
die "\nSys file $ARGV[1] is not well-formed" if $incorrect ;

print "GS:  $ARGV[0]\n" if $DBG ;
my $gs = loadalignments($ARGV[0],'gold') ;

print "SYS: $ARGV[1]\n" if $DBG ;
my $sys = loadalignments($ARGV[1],'sys') ;

printf " F1 Ali     %6.4f\n", F1($gs,$sys,'') ;
#printf " F1 Type    %6.4f\n", F1($gs,$sys,'type') ;
#printf " F1 Score   %6.4f\n", F1($gs,$sys,'score') ;
#printf " F1 Typ+Sco %6.4f\n", F1($gs,$sys,'typescore') ;


# the type of alignment only influences the following
#  - gs: store string of each token, remove if punctuation
#  - sys: use string of each token, remove if punctuation
sub loadalignments {
    my ($f,$type) = @_ ;
    my $alis = {} ;
    my ($id) ;
    my ($sent1,$sent2) ;
    open(I,$f) or die $! ;
    while (<I>) {
	chomp ;
	# extract pair id, and insert token strings in $alis if gold standard 
	if (/sentence id="([^\"]*)" /) {
	    $id = $1 ;
	    if ($type eq 'gold') {
		$sent1 = <I> ; chomp($sent1); $sent1 =~ s/^\/\/ //; $alis->{$id}{"string1"} = [ split(/ /,$sent1) ] ;
		$sent2 = <I> ; chomp($sent2); $sent2 =~ s/^\/\/ //; $alis->{$id}{"string2"} = [ split(/ /,$sent2) ] ;
	    }
	}

	# parse alignments
	if (/<==>/) {
	    die "contact developer" if not defined $id ;
	    next if not defined $id ;
	    # parse alignment 
	    my ($alignment,$types,$score,$comment) = split(/\/\//,$_) ;
	    my ($tokens1,$tokens2) = split(/<==>/,$alignment) ;
	    $tokens1 =~ s/^\s+// ; $tokens1 =~ s/\s+$// ; 
	    $tokens2 =~ s/^\s+// ; $tokens2 =~ s/\s+$// ; 
	    $score =~ s/^\s+// ; $score =~ s/\s+$// ; 
	    $types =~ s/^\s+// ; $types =~ s/\s+$// ; 
	    my @tokens1 =  split(/\s+/,$tokens1) ;
	    my @tokens2 =  split(/\s+/,$tokens2) ;
	    my @types =    split(/_/,$types) ;

	    # store chunk alignments, including NOALI
	    $alis->{$id}{"segments12"}{$tokens1}{$tokens2} = [ @types ] ;
	    $alis->{$id}{"segments21"}{$tokens2}{$tokens1} = [ @types ] ;

	    # produce token:token alignments, unless NOALI or ALIC (where one chunk is null, represented by 0)
	    next if $tokens1[0] == 0 ;  
	    next if $tokens2[0] == 0 ;   

	    # remove punctuation from evaluation
	    my $tmp ;
	    if ($type eq 'gold'){ $tmp = $alis } else { $tmp = $gs } ;
	    @tokens1 = grep { $tmp->{$id}{"string1"}[$_-1] !~ /^[.,:\'\`?;\"-]$/} @tokens1 ;
	    @tokens2 = grep { $tmp->{$id}{"string2"}[$_-1] !~ /^[.,:\'\`?;\"-]$/} @tokens2 ;
	    next if ! @tokens1 ;
	    next if ! @tokens2 ;

	    # produce token:token alignments and index them by token and by alignmet in both directions 
	    foreach my $t1 (@tokens1) {
		foreach my $t2 (@tokens2) {
		    # store @type separately for all kinds of alignments
		    $alis->{$id}{"tokens12"}{$t1}{$t2} = [ @types ] ;
		    $alis->{$id}{"tokens21"}{$t2}{$t1} = [ @types ] ;
		    $alis->{$id}{"links12"}{"$t1 $t2"} = [ @types ] ;
		    $alis->{$id}{"links21"}{"$t2 $t1"} = [ @types ] ;
		    # store $score separately for all kinds of alignments
		    $alis->{$id}{"tokens12score"}{$t1}{$t2} = $score ;
		    $alis->{$id}{"tokens21score"}{$t2}{$t1} = $score ;
		    $alis->{$id}{"links12score"}{"$t1 $t2"} = $score ;
		    $alis->{$id}{"links21score"}{"$t2 $t1"} = $score ;
		}
	    }
	}
    }
    die "No alignments found in $f, terminating" if (scalar keys %$alis) == 0 ;
    print "   Number of pairs $f: " . (scalar keys %$alis) . "\n" if $DBG ;
    return $alis ;
}


# check set equality for types
sub equalset {
    my ($set1,$set2) = @_ ;
    my $hash1 = { map { ($_, 1) } @$set1 } ;  
    my $hash2 = { map { ($_, 1) } @$set2 } ;  
    my $equal = 1 ;
    foreach my $el1 (keys %$hash1) {
	if (! $hash2->{$el1} ) { $equal = 0 ; last ; } ;
    }
    foreach my $el2 (keys %$hash2) {
	if (! $hash1->{$el2} ) { $equal = 0 ; last ; } ;
    }
    print ("   type mismatch: " . join("_",@$set1) . " and " . join("_",@$set2). "\n") if $DBG and ! $equal ;
    return $equal ;
}

# check jaccard between sets of types, normalizing to lowercase
sub jaccardset {
    my ($set1,$set2) = @_ ;
    my $hash1 = { map { (lc $_, 1) } @$set1 } ;  
    my $hash2 = { map { (lc $_, 1) } @$set2 } ;  
    my $intersect = {} ;
    my $union = {} ;
    foreach my $el1 (keys %$hash1) {
	$union->{$el1} = 1 ;
	$intersect->{$el1} = 1 if $hash2->{$el1} ;
    }
    foreach my $el2 (keys %$hash2) {
	$union->{$el2} = 1 ;
	$intersect->{$el2} = 1 if $hash1->{$el2} ;
    }
    my $jaccard = scalar(keys %$intersect) / scalar(keys %$union) ;
    return $jaccard ;
}



# check jaccard between sets of types, normalizing to lowercase
# no penalty between tags (SPE1, SPE2, REL, SIMI) when score is (0-2] (F1AST)
# no penalty between EQUI and SIMI/SPE with score 4 (F1AST)
# add 1 to intersection, substract 1 to union
sub jaccardsetNOP {
    my ($set1,$set2) = @_  ;		    
    my $hash1 = { map { (lc $_, 1) } @$set1 } ;  
    my $hash2 = { map { (lc $_, 1) } @$set2 } ;  
    my $intersect = {} ;
    my $union = {} ;
    foreach my $el1 (keys %$hash1) {
	$union->{$el1} = 1 ;
	$intersect->{$el1} = 1 if $hash2->{$el1} ;
    }
    foreach my $el2 (keys %$hash2) {
	$union->{$el2} = 1 ;
	$intersect->{$el2} = 1 if $hash1->{$el2} ;
    }
    my $jaccard = (1 + scalar(keys %$intersect)) / (-1 + scalar(keys %$union)) ;
    return $jaccard ;
}


# intersection between two sets
sub intersect {
    my ($set1,$set2) = @_ ;
    my $hash1 = { map { (lc $_, 1) } @$set1 } ;  
    my $hash2 = { map { (lc $_, 1) } @$set2 } ;  
    my $intersect = {} ;
    foreach my $el1 (keys %$hash1) {
	$intersect->{$el1} = 1 if $hash2->{$el1} ;
    }
    foreach my $el2 (keys %$hash2) {
	$intersect->{$el2} = 1 if $hash1->{$el2} ;
    }
    return [ keys %$intersect ] ;
}

sub EQUI {
    my ($typeset)=@_ ;
    foreach my $type (@$typeset) {
	return 1 if $type =~ /^EQUI/i ;
    }
    return 0 ;
}

sub SIMISPE {
    my ($typeset)=@_ ;
    foreach my $type (@$typeset) {
	return 1 if $type =~ /^(SIMI|SPE)/i ;
    }
    return 0 ;
}

sub SIMISPEREL {
    my ($typeset)=@_ ;
    foreach my $type (@$typeset) {
	return 1 if $type =~ /^(SIMI|SPE|REL)/i ;
    }
    return 0 ;
}

# equation 4 needs fanout, the number of token:token alignments per token 
#      [guard]
#      [death camp guard]
#
# e.g. fanout($alis->{"1"}{"tokens12"}{"guard"} = { death => [ equi ], 
#                                                   camp => [ equi ], 
#                                                   guard => [ equi ]} )
#         = 3 ;
# e.g. fanout($alis->{"1"}{"tokens21"}{"guard"} = { death => [ equi ]} )
#         = 1 ;

# number of token-token alignments for a given token in a given alignment direction
sub fanout {
    my ($ali) = @_ ;
    return scalar(keys %$ali) ;
}

# summatory of fan-out factors for all token-token alignments ( eq. 4)
sub countFanOut {
    my ($ali) = @_ ;
    my $count = 0;
    foreach my $token1 (keys %{ $ali->{"tokens12"}}) {
	foreach my $token2 (keys %{ $ali->{"tokens12"}{$token1}}) {
	    $count+= 1/max(fanout($ali->{"tokens12"}{$token1}),fanout($ali->{"tokens21"}{$token2})) ;
	}
    }
    return $count ; 
}

# Main function
sub F1 {
    my ($alisgs,$alissys,$mode) = @_ ;
    my $overlapGS ;
    my $overlapSYS ;
    my $linkssys ;
    print "\n F1 $mode ========\n" if $DBG ;
    print "    recall per pair (gs, sys)\n" if $DBG ;
    foreach my $id (sort {$a <=> $b} keys %$alissys) {
	$linkssys += countFanOut($alissys->{$id}) ; 
	next if not $alisgs->{$id} ;
	$overlapSYS += overlap($alissys->{$id},$alisgs->{$id},$id,$mode) ;
    }
    my $linksgs ;
    print "    precision per pair (sys, gs)\n" if $DBG ;
    foreach my $id (sort {$a <=> $b} keys %$alisgs) {
	$linksgs += countFanOut($alisgs->{$id}) ; 
	next if not $alissys->{$id} ;
	$overlapGS += overlap($alisgs->{$id},$alissys->{$id},$id,$mode) ;
    }
    my $precision = ($linkssys == 0) ? 0 : $overlapSYS / $linkssys ;
    my $recall =    ($linksgs == 0) ? 0 : $overlapGS / $linksgs ;
    my $f1 =        ($precision + $recall == 0) ? 0 : 2*$precision*$recall/($precision + $recall) ;
    print  " F1 overlapSYS:   $overlapSYS\n" if $DBG ;
    print  " F1 system links: $linkssys\n" if $DBG ;
    printf " F1 precision:    %4.2f\n",$precision if $DBG ;
    print  " F1 overlapGS:    $overlapGS\n" if $DBG ;
    print  " F1 gs links:     $linksgs\n" if $DBG ;
    printf " F1 recall:       %4.2f\n",$recall if $DBG ;
    return $f1 ;
}

# Fuzzy overlap using fanout from first alignment set
sub overlap {
    my ($ali1,$ali2,$id,$mode) = @_ ;
    my $overlap = 0;
    foreach my $token1 (keys %{ $ali1->{"tokens12"}}) {
	foreach my $token2 (keys %{ $ali1->{"tokens12"}{$token1}}) {
	    if ($mode eq "") {
		$overlap+= 1/max(fanout($ali1->{"tokens12"}{$token1}),fanout($ali1->{"tokens21"}{$token2}))
		    if $ali2->{"tokens12"}{$token1}{$token2} ;
	    } elsif ($mode eq "type") {
		if ($ali2->{"tokens12"}{$token1}{$token2}) { 		    # and equalset($ali1->{"tokens12"}{$token1}{$token2},$ali2->{"tokens12"}{$token1}{$token2}) ;
		    my $jaccard = jaccardset($ali1->{"tokens12"}{$token1}{$token2},$ali2->{"tokens12"}{$token1}{$token2}) ;
		    print ("        type mismatch: $jaccard " . join("_",@{$ali1->{"tokens12"}{$token1}{$token2}}) . " and " . join("_",@{$ali2->{"tokens12"}{$token1}{$token2}}). "\n") if $DBG and $jaccard != 1;
		    $overlap+= 1/max(fanout($ali1->{"tokens12"}{$token1}),fanout($ali1->{"tokens21"}{$token2}))
			* $jaccard ;
		}
	    } elsif ($mode eq "score") {
		$overlap+= 1/max(fanout($ali1->{"tokens12"}{$token1}),fanout($ali1->{"tokens21"}{$token2}))
		    * (1 - abs($ali1->{"tokens12score"}{$token1}{$token2} - $ali2->{"tokens12score"}{$token1}{$token2}) / 5)  
		    if $ali2->{"tokens12"}{$token1}{$token2} ;
	    } elsif ($mode eq "typescore") {
		next if not $ali2->{"tokens12"}{$token1}{$token2} ;
		my $typeset1 = $ali1->{"tokens12"}{$token1}{$token2} ;
		my $typeset2 = $ali2->{"tokens12"}{$token1}{$token2} ;
		my $score1 = $ali1->{"tokens12score"}{$token1}{$token2};
		my $score2 = $ali2->{"tokens12score"}{$token1}{$token2};
		my $overlapincrease = 1/max(fanout($ali1->{"tokens12"}{$token1}),fanout($ali1->{"tokens21"}{$token2})) 
		    * (1 - abs($score1 - $score2) / 5)  ;
		my $jaccard ;
		if ((EQUI($typeset1) and SIMISPE($typeset2) and ($score2>=4)) 
		    or (EQUI($typeset2) and SIMISPE($typeset1) and ($score1>=4))) {
                    # no type penalty between EQUI and SIMI/SPE with score 4 (F1AST)
		    $jaccard = jaccardset($typeset1,$typeset2) ;		    # jaccardsetNOP($typeset1,$typeset2) ;		    
		}
		elsif (SIMISPEREL($typeset1) and SIMISPEREL($typeset2) and not(SIMISPEREL(intersect($typeset1,$typeset2)))
		       and ($score1 < 3) and  ($score2 < 3)) {
		    # no type penalty between tags (SPE1, SPE2, REL, SIMI) when score is (0-2] (F1AST)
                    # and ali1 and ali2 have different type.
		    $jaccard = jaccardset($typeset1,$typeset2) ; # jaccardsetNOP($typeset1,$typeset2) ;
		}
		else { # standard match
		    $jaccard = jaccardset($typeset1,$typeset2) ;
		    # and equalset($ali1->{"tokens12"}{$token1}{$token2},$ali2->{"tokens12"}{$token1}{$token2}) ;
		}
		print ("        type mismatch: $jaccard " . join("_",@{$typeset1}) . " ($score1) and " . join("_",@{$typeset2}). " ($score2)\n") if $DBG and $jaccard != 1;
		$overlap+= $overlapincrease*$jaccard ;
	    } else { die }
	}
    }
    if ($DBG and $id) {
	my ($total,$precORrecall) ;
	printf "   " ;
	printf "%2d: ",$id ;
	$total = countFanOut($ali1) ; 
        if ($total) {
	    $precORrecall = $overlap / $total ;
	} else {
	    $precORrecall = 0 ;
	}
	printf "ov:%5.2f tot:%5.2f p/r:%4.2f\n",$overlap,$total,$precORrecall ;
    }
    return $overlap ; 
}
	    