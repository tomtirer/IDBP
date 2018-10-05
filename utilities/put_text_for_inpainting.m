function [img_with_text,deletion_set] = put_text_for_inpainting(img)

[M,N] = size(img);

font_type = 'Times New Roman Bold';
font_size = 30;
hor_margin = 2;
ver_margin = 3;
ver_gap = 35;


temp_with_text = ones(M,N);
text_str = 'O Romeo, Romeo, wherefore art thou Romeo?';
sum_gap = ver_margin;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'Deny thy father and refuse thy name';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'Or if thou wilt not, be but sworn my love';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'And I’ll no longer be a Capulet.';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = '‘Tis but thy name that is my enemy:';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'Thou art thyself, though not a Montague.';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'What’s Montague? It is nor hand nor foot';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'Nor arm nor face nor any other part';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'Belonging to a man. O be some other name.';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'What’s in a name? That which we call a rose';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'By any other name would smell as sweet;';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'So Romeo would, were he not Romeo call’d,';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'Retain that dear perfection which he owes';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'Without that title. Romeo, doff thy name,';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'And for that name, which is no part of thee,';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');
text_str = 'Take all myself.';
sum_gap = sum_gap+ver_gap;
temp_with_text = insertText(temp_with_text,[hor_margin,sum_gap],text_str,'BoxOpacity',1,'FontSize',font_size,'Font',font_type,'BoxColor','white');


img_text = temp_with_text(:,:,1)>0.85;
deletion_set = find(img_text==0);
img_with_text = img;
img_with_text(deletion_set) = 0;


