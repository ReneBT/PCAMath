load('FA profile data.jrb','-mat')
load('D:\Algo\Jupyter\PCA\PCAMath.git\trunk\data\FA spectra.mat','-mat')
spectra.molarprofile = bsxfun(@rdivide, ANNSBUTTERmass, FAproperties.MolarMass);
spectra.molarprofile = 100.*bsxfun(@rdivide, spectra.molarprofile, sum(spectra.molarprofile));
spectra.Butter = simFA*spectra.molarprofile;
tpcs = 11;
Row_D = spectra.Butter';
[Row_LE,Row_sv,Row_RE] = svd(Row_D,'econ');
Row_LE(:,tpcs:end) = [];
Row_sv(tpcs:end,:) = [];
Row_sv(:,tpcs:end) = [];
Row_RE(:,tpcs:end) = [];
Row_sc = Row_LE*Row_sv;
Data_Row_Recon = Row_sc*Row_RE';
%try calculating the scores
pIRow_RE = pinv(Row_RE);
Row_sc_xL = Row_D*Row_RE;
Row_sc_pInv = (pIRow_RE*Row_D')';
Row_L_DTSiT = Row_D'*pinv(Row_sc)'; %DtSiT pn nd
Row_L_SiD = (pinv(Row_sc)*Row_D)';

Column_D = spectra.Butter;
[Column_LE,Column_sv,Column_RE] = svd(Column_D,'econ');
Column_LE(:,tpcs:end) = [];
Column_sv(tpcs:end,:) = [];
Column_sv(:,tpcs:end) = [];
Column_RE(:,tpcs:end) = [];
Column_sc = Column_sv*Column_RE';
Data_Column_Recon = Column_LE*Column_sc;
%try calculating the scores
Column_piLE = pinv(Column_LE);
Column_sc_xL = (Column_D'*Column_LE)';
Column_sc_pInv = Column_piLE*Column_D;
Column_L_DSi = Column_D*pinv(Column_sc); %DtSiT pn nd
Column_LT_SiD = pinv(Column_sc)'*Column_D';

matrix_Dims = [size(spectra.Butter'), size(Row_sc), size(Row_RE);...
    size(spectra.Butter), size(Column_sc), size(Column_RE)];