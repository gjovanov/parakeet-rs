For all .wav files in folder ./media e.g (broadcast_1.wav) we already habe their acompanying transcript (e.g. broadcast_1.transcript.txt).

Our goal is to create live (realtime) transcription using using webrtc_transcriber, canary model and speedy mode, that will transcribe audio in smaller chunks and generate growing sentences/segments e.g.


Wort zum Thema
Wort zum Thema Ortszentren
Wort zum Thema Ortszentren lädt der ORF
Wort zum Thema Ortszentren lädt der ORF am 19. November
Wort zum Thema Ortszentren lädt der ORF am 19. November nach Bischofshofen.

Eine Anmeldung
Eine Anmeldung ist erforderlich.

Dann ist das vielleicht
Dann ist das vielleicht die beste Pointe
Dann ist das vielleicht die beste Pointe des Abends.

Manuel Hubei
Manuel Hubei und Simon Schwarz
Manuel Hubei und Simon Schwarz spielen ihr Programm
Manuel Hubei und Simon Schwarz spielen ihr Programm Das Restaurant
Manuel Hubei und Simon Schwarz spielen ihr Programm Das Restaurant am 17. November
Manuel Hubei und Simon Schwarz spielen ihr Programm Das Restaurant am 17. November im Kulturzentrum Hallwang.

90 Jahre Società
90 Jahre Società Tante Alighieri
90 Jahre Società Tante Alighieri werden mit einem Festa Bertutti
90 Jahre Società Tante Alighieri werden mit einem Festa Bertutti am 15. November
90 Jahre Società Tante Alighieri werden mit einem Festa Bertutti am 15. November im Carabinieri-Saal 
90 Jahre Società Tante Alighieri werden mit einem Festa Bertutti am 15. November im Carabinieri-Saal im Domquartier gefeiert.


In the frontend in the html selement with this css selector:
a) section#live-subtitle-section > div#live-subtitle > div#subtitle-text, we should display always the latest/current 
 -> e.g. 90 Jahre Società Tante Alighieri werden mit einem Festa Bertutti am 15. November im Carabinieri-Saal im Domquartier gefeiert. 
b) list of all the above segments in reversed order (latest on the top).

Analyze all ./media/*.wav files (for max 5 mins) and their existing .txt transcript and propose a solution in both backend and frontend to achive the growing sentences/segments