%%
clc
clear
clf
close all;

data = load('DTMF_noisy_signal.mat');

%Load
f_sample = data.f_sample;
samples = data.s_total;
N = length(samples);
freq_res = f_sample/N;

%Tid
x= linspace(1,N/f_sample,N);

%Fjernelse af DC offset
DC_offset = mean(samples);
samples_noOffset = samples - DC_offset;

%Signal
figure(1);
plot(x,samples, 'r');
hold on
plot(x,samples_noOffset, 'b');
title('Originalt signal');
legend('Originalt signal', 'Uden DC offset')


%%


%FFT af original
ft_samples = fft(samples_noOffset);
num_bins = N/2;

figure(2);
subplot(3,1,1);
plot([0:num_bins*2-1], ft_samples(1:num_bins*2));
title('FFT signal med spejling');

%FFT af originalt signal i bins
subplot(3,1,2);
plot([0:num_bins-1], ft_samples(1:num_bins));
title('FFT af originalt signal i bins');

%FFT af originalt signal i Hz
subplot(3,1,3);
plot([0:num_bins-1].*freq_res, ft_samples(1:num_bins));
title('FFT af orginalt signal i Hz');





%Her påbegyndes vinduesmetoden. Der ønskes et båndpasfilter med følgende
%cutoff-frekvenser:
f_cutoff1 = 690;
f_cutoff2 = 1640;

%Frekvensernes tilsvarende bins findes
freq_bin1 = f_cutoff1 / freq_res;
freq_bin_round1 = round(freq_bin1);
freq_bin2 = f_cutoff2 / freq_res;
freq_bin_round2 = round(freq_bin2);

%Der laves 1'ere hvor vi gerne vil bibeholde frekvenser, og 0'ere hvor vi
%ikke vil
H_left  = [zeros(1,freq_bin_round1-1) ones(1,freq_bin_round2-freq_bin_round1+1) zeros(1,(N/2)-freq_bin_round2)];

%Spejling
H_right = fliplr(H_left(1:end));

%Samlet vindue
H = [H_left H_right];

%Vindue ganges på FFT af samples
ft_samples_BP = ft_samples.*H;

%Plot ft_samples_BP
figure(3); clf;
plot([0:num_bins-1].*freq_res, ft_samples(1:num_bins), 'r');
hold on
plot([0:num_bins-1].*freq_res, ft_samples_BP(1:num_bins));
title('FFT af orginalt signal i Hz');

%Signal efter de to filtre
samples_BP = ifft(ft_samples_BP);

figure(4); clf;
plot(x, samples_noOffset, 'r');
hold on
plot(x, samples_BP);

%Signal sendes gennem stopbåndsfiltre, der fjerner de uønskede frekvenser
%mellem DTMF-tonerne
filtered1 = StopBandFunctionDTMF(f_sample,samples_BP,710,750,3);
filtered2 = StopBandFunctionDTMF(f_sample,filtered1,790,830,3);
filtered3 = StopBandFunctionDTMF(f_sample,filtered2,870,920,3);
filtered4 = StopBandFunctionDTMF(f_sample,filtered3,960,1190,3);
filtered5 = StopBandFunctionDTMF(f_sample,filtered4,1230,1315,3);
filtered6 = StopBandFunctionDTMF(f_sample,filtered5,1355,1455,3);
filtered7 = StopBandFunctionDTMF(f_sample,filtered6,1495,1610,3);

samples_BP_STOP = filtered7;

figure(5); clf;
plot(x, samples_BP_STOP);


stop_ft_mags = abs(fft(filtered7));


% 
% cutoff_stop_low = 950/freq_resolution;
% cutoff_stop_high = 1150/freq_resolution;
% 
% cutoff_stop_low_nyquist = cutoff_stop_low/num_bins_to_display;
% cutoff_stop_high_nyquist = cutoff_stop_high/num_bins_to_display;
% 
% [b,a] = butter(3,[cutoff_stop_low_nyquist cutoff_stop_high_nyquist], 'stop');
% 
% stop_samples = filter(b,a,samples);
% 
% stop_ft_mags = abs(fft(stop_samples));


figure(6); clf;
plot([0:num_bins-1].*freq_res, stop_ft_mags(1:num_bins));



% nyquist_f = f_sample/2;
% 
% ftype = 'stop';
% n = 6;
% cutoff_low = 1000;
% cutoff_high = 1100;
% 
% cutoff_low_nyquist= cutoff_low/nyquist_f;
% cutoff_high_nyquist = cutoff_high/nyquist_f;
% 
% [b,a] = butter(n,[cutoff_low_nyquist, cutoff_high_nyquist],ftype);



% h_1 = ifft(H);
% h = fftshift(real(ifft(H)));
% w_hanning = hanning(N)';
% h_win = h.*w_hanning;
% 
% %Frequency spectrum with bandpass and hanning
% ft_mags_BP_hanning = ft_mags.*h_win;
% samples_BP_hanning = ifft(ft_mags_BP_hanning);
% 
% 
% H_without_win = fft(h,f_sample);
% H_with_win = fft(h_win,f_sample);

% figure(4); clf
% stem(H_left)
% figure(5); clf
% plot(h)
% axis([1 N-1 -inf inf])
% hold on
% plot(w_hanning*max(abs(h)),'g','linewidth',2)
% plot(h_win,'r','linewidth',2)
% title('Impulsrespons med vinduesfunktion')
% 
% figure(6); clf
% plot(abs(H_without_win(1:f_sample/2)))
% hold on
% plot(abs(H_with_win(1:f_sample/2)),'r','linewidth',2)
% grid on
% title('Resulterende overføringsfunktion')




%%
w_samples = samples.*win;


w_samples_fft = fft(w_samples)/N;


w_samples_fft_noDC = fft(w_samples-mean(w_samples).*win)/N;


figure(1);
%hold on;
plot(20*log10(abs(w_samples_fft)),'b');

figure(2);
freqz(w_samples_fft);


figure(3);
%hold on;
plot(20*log10(abs(w_samples_fft_noDC)),'b');

figure(4);
plot(20*log10(abs(samples_fft)),'r');