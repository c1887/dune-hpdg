// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// // vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_HPDG_LOCALFUNCTIONS_GAUSS_LOBATTO_POINTS_HH
#define DUNE_HPDG_LOCALFUNCTIONS_GAUSS_LOBATTO_POINTS_HH

#include <dune/common/fvector.hh>
namespace Dune 
{
	template<int k>
  class GaussLobattoPoints
	{
      public:

        static FieldVector<double, k> getPoints() {
            FieldVector<double, k> points;
			switch(k) {
            case 0:
            case 1:
                DUNE_THROW(Dune::NotImplemented, "No GL points for order 0");
            case 2:
                points[0] = 0.0;
                points[1] = 1.0;
                return points;
            case 3:
                points[0] = 0.0;
                points[1] = 0.5;
                points[2] = 1.0;
                return points;
            case 4:
                points[0] = 0.0;
                points[1] = 0.27639320225;
                points[2] = 0.72360679775;
                points[3] = 1.0;
                return points;
            case 5:
                points[0] = 0.0;
                points[1] = 0.172673164646;
                points[2] = 0.5;
                points[3] = 0.827326835354;
                points[4] = 1.0;
                return points;
            case 6:
                points[0] = 0.0;
                points[1] = 0.117472338035;
                points[2] = 0.35738424176;
                points[3] = 0.64261575824;
                points[4] = 0.882527661965;
                points[5] = 1.0;
                return points;
            case 7:
                points[0] = 2.90444393614e-121;
                points[1] = 0.0848880518607;
                points[2] = 0.265575603265;
                points[3] = 0.5;
                points[4] = 0.734424396735;
                points[5] = 0.915111948139;
                points[6] = 1.0;
                return points;
            case 8:
                points[0] = 1.54903676594e-120;
                points[1] = 0.0641299257452;
                points[2] = 0.204149909283;
                points[3] = 0.395350391049;
                points[4] = 0.604649608951;
                points[5] = 0.795850090717;
                points[6] = 0.935870074255;
                points[7] = 1.0;
                return points;
            case 9:
                points[0] = 0.0;
                points[1] = 0.0501210022943;
                points[2] = 0.161406860245;
                points[3] = 0.318441268087;
                points[4] = 0.5;
                points[5] = 0.681558731913;
                points[6] = 0.838593139755;
                points[7] = 0.949878997706;
                points[8] = 1.0;
                return points;
            case 10:
                points[0] = 0.0;
                points[1] = 0.0402330459168;
                points[2] = 0.130613067447;
                points[3] = 0.261037525095;
                points[4] = 0.417360521167;
                points[5] = 0.582639478833;
                points[6] = 0.738962474905;
                points[7] = 0.869386932553;
                points[8] = 0.959766954083;
                points[9] = 1.0;
                return points;
            case 11:
                points[0] = 0.0;
                points[1] = 0.032999284796;
                points[2] = 0.107758263168;
                points[3] = 0.217382336502;
                points[4] = 0.352120932207;
                points[5] = 0.5;
                points[6] = 0.647879067793;
                points[7] = 0.782617663498;
                points[8] = 0.892241736832;
                points[9] = 0.967000715204;
                points[10] = 1.0;
                return points;
            case 12:
                points[0] = 1.78139228083e-119;
                points[1] = 0.0275503638886;
                points[2] = 0.090360339178;
                points[3] = 0.183561923484;
                points[4] = 0.300234529517;
                points[5] = 0.431723533573;
                points[6] = 0.568276466427;
                points[7] = 0.699765470483;
                points[8] = 0.816438076516;
                points[9] = 0.909639660822;
                points[10] = 0.972449636111;
                points[11] = 1.0;
                return points;
            case 13:
                points[0] = 0.0;
                points[1] = 0.0233450766789;
                points[2] = 0.0768262176741;
                points[3] = 0.156905765459;
                points[4] = 0.258545089454;
                points[5] = 0.375356534947;
                points[6] = 0.5;
                points[7] = 0.624643465053;
                points[8] = 0.741454910546;
                points[9] = 0.843094234541;
                points[10] = 0.923173782326;
                points[11] = 0.976654923321;
                points[12] = 1.0;
                return points;
            case 14:
                points[0] = 5.80888787227e-121;
                points[1] = 0.0200324773664;
                points[2] = 0.0660994730848;
                points[3] = 0.135565700454;
                points[4] = 0.224680298536;
                points[5] = 0.328637993329;
                points[6] = 0.441834065558;
                points[7] = 0.558165934442;
                points[8] = 0.671362006671;
                points[9] = 0.775319701464;
                points[10] = 0.864434299546;
                points[11] = 0.933900526915;
                points[12] = 0.979967522634;
                points[13] = 1.0;
                return points;
            case 15:
                points[0] = 1.77655154094e-118;
                points[1] = 0.0173770367481;
                points[2] = 0.0574589778885;
                points[3] = 0.118240155024;
                points[4] = 0.196873397265;
                points[5] = 0.289680972643;
                points[6] = 0.392323022318;
                points[7] = 0.5;
                points[8] = 0.607676977682;
                points[9] = 0.710319027357;
                points[10] = 0.803126602735;
                points[11] = 0.881759844976;
                points[12] = 0.942541022111;
                points[13] = 0.982622963252;
                points[14] = 1.0;
                return points;
            case 16:
                points[0] = 1.71749451424e-118;
                points[1] = 0.0152159768649;
                points[2] = 0.0503997334533;
                points[3] = 0.103995854069;
                points[4] = 0.173805648559;
                points[5] = 0.256970289056;
                points[6] = 0.35008476555;
                points[7] = 0.449336863239;
                points[8] = 0.550663136761;
                points[9] = 0.64991523445;
                points[10] = 0.743029710944;
                points[11] = 0.826194351441;
                points[12] = 0.896004145931;
                points[13] = 0.949600266547;
                points[14] = 0.984784023135;
                points[15] = 1.0;
                return points;
            case 17:
                points[0] = 0.0;
                points[1] = 0.0134339116843;
                points[2] = 0.0445600020422;
                points[3] = 0.0921518743891;
                points[4] = 0.154485509686;
                points[5] = 0.229307300335;
                points[6] = 0.313912783217;
                points[7] = 0.405244013241;
                points[8] = 0.5;
                points[9] = 0.594755986759;
                points[10] = 0.686087216783;
                points[11] = 0.770692699665;
                points[12] = 0.845514490314;
                points[13] = 0.907848125611;
                points[14] = 0.955439997958;
                points[15] = 0.986566088316;
                points[16] = 1.0;
                return points;
            default:
                DUNE_THROW(Dune::NotImplemented, "no more Gauss Lobatto rules for k+1>17");
			}
		}
	};
}
#endif
