function myplayer(s,t,fs)
    player = audioplayer(s,fs);
    play(player)
    pause(t);
    stop(player);
end