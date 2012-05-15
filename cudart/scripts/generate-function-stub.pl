#!/usr/bin/perl -w

use strict;
use File::Basename;
use File::Slurp;
use Getopt::Long qw(:config bundling no_ignore_case);

$|=1;

my %cfg = (
	    functions => [],
	    file      => '',
	    include   => '/usr/local/cuda/include/',

	    write     => 0,
	    code      => 'cudart.c',
	    spec      => 'cudart.dll.spec',

	    dry_run   => 0,
	    verbose   => 0,
	    help      => 0,
	  );

my $result = GetOptions(
			 'f|function=s@' => \@{$cfg{functions}},
			 'F|file=s'      => \$cfg{file},
			 'i|include=s'   => \$cfg{include},
			 'c|code=s'      => \$cfg{code},
			 's|spec=s'      => \$cfg{spec},

			 'n|dry-run'     => \$cfg{dry_run},
			 'v|verbose'     => \$cfg{verbose},
			 'h|help'        => \$cfg{help},
		       );

if(!$result || $cfg{help} || (!@{$cfg{functions}} && !$cfg{file})) {
	print STDERR "\n" if(!$result);

	print "Error: No functions nor file specified!\n\n" if(!@{$cfg{functions}} && !$cfg{file});

	print "Usage: ". basename($0) ." [OPTIONS]\n\n";
	print "Options:\n";
	print "-f, --function <NAME>  Generate stub for specified function(s)\n";
	print "-F, --file <PATH>      Generate stub for function(s) in specified file\n";
	print "-i, --include <PATH>   Path to CUDA include files ($cfg{include})\n";
	print "\n";
	print "-w, --write            Append generated code to files\n";
	print "-c, --code <PATH>      Path to cudart.c           ($cfg{code})\n";
	print "-s, --spec <PATH>      Path to cudart.dll.spec    ($cfg{spec})\n";
	print "\n";
	print "-n, --dry-run          Don't write modified files\n";
	print "-v, --verbose          Enable verbose output\n";
	print "-h, --help             Display this usage information\n";

	exit 1;
}

my %cuda = (
	     runtime_api => {
		              file => 'cuda_runtime_api.h',
			      data => '',
			    },
	   );

foreach my $key (sort keys %cuda) {
	my $file  = $cfg{include};
	   $file .= '/' if(substr($cfg{include}, -1, 1) ne '/');
	   $file .= $cuda{$key}{file};

	print "Loading file: $file\n" if($cfg{verbose});

	if(!-r $file) {
		print "Error: Cannot read file: $file\n";
		exit 1;
	}

	my @data = read_file($file);

	$cuda{$key}{data} = \@data;
}

my $code = '';
my $spec = '';

if($cfg{file} && -r $cfg{file}) {
	foreach my $function (read_file($cfg{file})) {
		chomp($function);

		my ($c, $s) = generate_function($function);

		$code .= $c;
		$spec .= $s;
	}
}
elsif($cfg{file} && !-r $cfg{file}) {
	print "Error: Cannot read file: $cfg{file}\n";
	exit 1;
}

foreach my $function (@{$cfg{functions}}) {
	my ($c, $s) = generate_function($function);
	
	$code .= $c;
	$spec .= $s;
}

print $code;
print "\n\n";
print $spec;

if($cfg{write} && !$cfg{dry_run}) {
	print "Appending functions to code and spec files\n" if($cfg{verbose});

	if($cfg{code} && -w $cfg{code}) {
		write_file($cfg{code}, { append => 1 }, $code);
	}
	if($cfg{spec} && -w $cfg{spec}) {
		write_file($cfg{spec}, { append => 1 }, $spec);
	}
}
elsif($cfg{write}) {
	print "Not appending functions to code and spec files (DRY RUN)\n" if($cfg{verbose});
}

exit 0;

################################################################################
# Subroutines

sub generate_function {
	my ($function) = @_;

	print "Generating function: $function\n" if($cfg{verbose});

	my $code = '';
	my $spec = '';

	foreach(@{$cuda{runtime_api}{data}}) {
		# cuda_runtime_api.h:
		# 
		# extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
		#
		#
		# cudart.c:
		#
		# cudaError_t WINAPI wine_cudaMemcpy3DAsync( const struct cudaMemcpy3DParms *p, cudaStream_t stream ){
		#         WINE_TRACE("\n");
		#         return cudaMemcpy3DAsync( p, stream );
		# }
		# 
		#
		# cudart.dll.spec:
		#
		# @  stdcall cudaMemcpy3DAsync( long long ) wine_cudaMemcpy3DAsync
			
		if(/extern __host__ cudaError_t CUDARTAPI ($function\((.*?)\));/) {
			my $prototype = $1;
			my $parameter = $2;

			my $return = $function.'(';
			my $args   = '';

			# const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0)
			my @params = split /,/, $parameter;
			
			foreach my $param (@params) {
				my @words = split / /, $param;

				pop @words if($words[-1] eq '__dv(0)');

				last if($words[-1] eq 'void' && $#params == 0);

				my $ptr = 0;
				if($words[-1] =~ s/^\*+//) { $ptr = 1; }

				$return .= ', ' if(substr($return, -1, 1) ne '(');
				$return .= $words[-1];

				$args .= ' ' if($args ne '');
				if($ptr) {
					$args .= 'ptr';
				}
				else {
					$args .= 'long';
				}
			}

			$return .= ')';

			$code .= "cudaError_t WINAPI wine_${prototype}{\n";
			$code .= " " x 8;
			$code .= "WINE_TRACE(\"\\n\");\n";
			$code .= " " x 8;
			$code .= "return ${return};\n";
			$code .= "}\n\n";

			$spec .= "@  stdcall ${function}( $args ) wine_${function}\n";

			print "\n$code$spec\n" if($cfg{verbose});

			last;
		}
	}

	return ($code, $spec);
}
